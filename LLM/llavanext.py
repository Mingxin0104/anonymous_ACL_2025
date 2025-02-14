import os
import json
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse


model_name = ""
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def parse_args():
    parser = argparse.ArgumentParser(description="推理任务")
    parser.add_argument('--description_json_path', type=str, default='')
    parser.add_argument('--dataset_json_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()

def load_or_initialize_json(file_path):

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()  
                return json.loads(content) if content else {}
        except json.JSONDecodeError:
            print(f"Warning: JSON 文件 {file_path} 格式错误，将初始化为空字典。")
            return {}
    else:
        return {}

def main():
    args = parse_args()

    description_json_path = args.description_json_path
    dataset_json_path = args.dataset_json_path
    output_path = args.output_path
    with open(description_json_path, 'r', encoding='utf-8') as f:
        description_data = json.load(f)
    # 初始化输出 JSON 文件
    results = load_or_initialize_json(output_path)
    # 记录已经处理过的 video_name
    processed_video_names = set(results.keys())
    pattern = re.compile(r'/|\.mp4')
    result_pattern = re.compile(r'最终答案为\[([01])\]')
    for key, data in description_data.items():
        video_name = data.get('video_name')
        if not video_name:
            print(f"Warning: 未找到 video_name 对于键 {key}, 跳过...")
            continue
        if video_name in processed_video_names:
            continue
        pred = data.get('pred')
        if not pred:
            clean_video_id = pattern.sub('', video_name)
            print(f"Warning: Pred 为空对于视频 {clean_video_id}, 跳过...")
            continue
        # 使用正则表达式处理 video_name 得到 video_id
        video_id = pattern.sub('', video_name)
        title = None
        # 遍历第二个JSON文件找匹配的video_id及对应的title
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset_obj = json.loads(line.strip())
                    if dataset_obj.get("video_id") == video_id:
                        title = dataset_obj.get("title")
                        break
                except json.JSONDecodeError:
                    print(f"Warning: 无法解析行: {line}")
        if not title:
            clean_video_id = pattern.sub('', video_name)
            print(f"Warning: 未找到视频 {clean_video_id} 的 title, 跳过...")
            continue


        # 定义推理消息
        messages = [
            {"role": "system", "content": "你是一个内容理解与推理专家, 下面我给你一段定义:'请你根据输入的视频以及文本和不实信息或者虚假信息的定义，来判断当前输入的内容是否为不实信息或者虚假信息，若是不实信息或者虚假信息，请输出1，若不是，请输出0，以下是不实信息以及虚假信息的定义，其可以分为三类，1.失实新闻：指具有新闻事实根据，但却没有全面、正确、恰当报道新闻事实而形成的新闻。因此，失实新闻在性质上还属于新闻范畴，它是对一定新闻事实“残缺”、“偏离”、“片面”反映报道而成的新闻。正像“次品”相对“正品”那样，失实新闻是不同程度的“劣质”新闻，但不是“假” 新闻。2. 假新闻：以不实的“新闻事实”为依据而报道出来的“新闻”，即假新闻是没有任何客观事实根源的“新闻”。或者说，以虚构出的“新闻事实”为本源的“新闻”就是假新闻。3. 策划（性）新闻：由某些主体（组织、群体或个体，包括新闻媒体自身）故意策划、塑造甚至是制造出来的、专门“预备”给新闻报道的事实或事件，这类事实或事件可以统一称为“策划（性）事实或事件”，可简称为“策划事实”或“策划事件”，我们把对“策划事实”或“策划事件”报道后形成的新闻统一称为“策划性新闻”，可简称为“策划新闻”.',请你根据这一段定义进行推理输出结果."},
            {"role": "user", "content": f"请根据以下信息进行推理：Title: {title}. 若此信息为真则输出推理结果为[0], 反之则输出推理结果为[1]. 回答格式：最终答案为[]"},
            {"role": "user", "content": f"请根据我给你的内容一步一步考虑：1) 构建scene graph 2）给出最终的答案。The scenario is：{pred}"}
        ]

        # 模型输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # 推理
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        # 使用正则表达式匹配结果
        match = result_pattern.search(response)
        if match:
            result_value = match.group(1)
        else:
            print(f"Warning: 未在推理结果中匹配到 最终答案为[0] 或 最终答案为[1] 对于视频 {video_id}, 跳过保存结果。")
            continue
        # 保存推理结果
        results[video_id] = result_value
        print(f"处理视频 {video_id}: 保存结果 {result_value}")
        # 实时更新输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"所有推理结果已保存到 {output_path}")

if __name__ == "__main__":
    main()