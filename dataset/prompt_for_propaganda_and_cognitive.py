# coding=utf-8
import cv2  # 执行pip install opencv-python安装OpenCV模块
import base64
import time
from openai import OpenAI
import os
import requests
import json
from sklearn.metrics import classification_report
from datetime import datetime
import pytz  # 导入pytz库以处理时区问题
from volcenginesdkarkruntime import Ark
import pandas as pd


def get_beijing_time():
    """
    获取当前的北京时间。
    返回:
        datetime.datetime: 当前的北京时间。
    """
    beijing_tz = pytz.timezone('Asia/Shanghai')  # 创建北京时间（亚洲/上海）时区对象
    beijing_time = datetime.now(beijing_tz)  # 获取当前的北京时间
    return beijing_time


def extract_content_between_braces(text):
    # 找到第一个'{'的索引
    start_index = text.find('{')

    # 找到最后一个'}'的索引
    end_index = text.rfind('}')

    # 提取并返回两个索引之间的内容
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index + 1]  # 不包括大括号本身
    return None  # 如果没有找到或位置不正确则返回None


def save_data_to_json(json_data, output_path, encoding='utf-8'):
    # 将处理后的数据保存为一个JSON文件
    with open(output_path, 'w', encoding=encoding) as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)


start_time = time.time()

print()
print('*' * 100)
print(get_beijing_time())
print('*' * 100)
print()

model = 'gpt-4o-2024-11-20'
# model = 'gpt-3.5-turbo'


client = OpenAI(api_key=doubao_key, base_url=doubao_url)
# client = Ark(base_url=doubao_url, api_key=doubao_key)

# data = pd.read_json(input_path,orient='records',dtype=False,lines=True)
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)
# 当次循环遍历视频的数量
iter_video_num = 200
# 计数器
num = 0

# 错误数量
error_num = 0

# 遍历json列表
# for index, item in data.iterrows():
for item in data:
    if item['whisper'] == "":
        continue
    if num >= iter_video_num:
        break
    if item['whisper_bias_doubao'] != "":
        continue
    else:
        print(num)
        title = item['title']
        content = item['whisper']

        # 文宣手段检测的prompt
        PROMPT_TEMPLATE = """
        请对以下新闻文本进行全面分析，包括以下几个方面：

        识别使用的所有可能的宣传技术，并解释判断依据。
        分析这些宣传技术可能利用的认知偏差以及判断依据。
        确定新闻是否与中国相关。
        列出提及的组织和个人，以及他们对中国或中国政府的态度。

        请将分析结果以JSON格式输出，结构如下：
        {
            "propaganda_techniques": [
                {
                    "type": "宣传技术类型",
                    "text": "使用该宣传技术的文本",
                    "start": 起始位置,
                    "end": 结束位置,
                    "reason": "解释判断使用了该宣传手段的原因",
                    "cognitive_bias": {
                        "type": "可能利用的认知偏差类型",
                        "reason": "解释为什么判断使用了该认知偏差"
                    }
                }
            ],
            "attitude": {
                "China-related": true/false,
                "organisations": [
                    {
                        "org_name": "组织名称",
                        "attitude_to_China": "对中国的态度",
                        "attitude_towards_Chinese_government": "对中国政府态度",
                        "attitude_towards_Chinese_policies": "对中国政策态度",
                        "attitude_towards Chinese_leaders": "对中国领导人态度"
                    }
                ],
                "persons": [
                    {
                        "person_name": "人名",
                        "org_name": "所属组织",
                        "country": "国家",
                        "title": "职位",
                        "attitude_to_China": "对中国的态度",
                        "attitude_towards_Chinese_government": "对中国政府态度",
                        "attitude_towards_Chinese_policies": "对中国政策态度",
                        "attitude_towards Chinese_chairman": "对中国领导人态度"
                    }
                ]
            }
        }
        注意：

        宣传技术类型包括：
        1.加载语言（情绪语言）: 使用具有强烈情感暗示（正面或负面）的特定词语和短语来影响听众。
        2.贴标签：将宣传活动的对象标记为目标受众害怕、讨厌、不喜欢或喜欢、赞美的东西。
        3.诉诸反复：一遍又一遍地重复同样的信息，让观众最终接受。
        4.夸张（夸大或淡化）：要么以过度的方式表现某物：使事物变得更大、更好、更坏，或者使某物看起来不那么重要或比实际更小。
        5.诉诸质疑：质疑某人或某事的可信度。
        6.诉诸恐惧/偏见：尝试基于先入为主的判断，通过向人们灌输对替代方案的焦虑和/或恐慌来寻求对某个想法的支持。
        7.挥舞旗帜（高举大旗）：利用强烈的民族感情（或尊重任何群体，例如种族、性别、政治偏好）来证明或促进行动或想法，文字上彰显某些（普世）价值或愿景
        8.简化因果：当一个问题背后有多个原因时，假设一个原因或原因。简化因果还包括寻找替罪羊这种形式，即在不调查问题的复杂性的情况下将责任转移到一个人或一群人身上。
        9.	喊口号：一个简短而引人注目的短语，可能包括了贴标签与刻板印象。
        10.	诉诸权威：在没有任何其他支持证据的情况下，仅仅因为该问题有权威或专家支持它而声明该主张是真实的。
        11.	非黑即白：将两个替代选项作为唯一的可能性，而实际上存在更多可能性。
        12.	格言论证：阻碍对某个主题进行批判性思考和有意义讨论的词语或短语。 
        13.	诉诸转移：包括(a)什么主义（Whataboutism）：通过指责对手虚伪而不直接反驳他们的论点来诋毁他们的立场、(b)稻草人：通过偷换概念、歪曲原意、以偏概全等曲解对方的论点，针对曲解后的论点（替身稻草人）攻击，再宣称已推翻对方论点的论证方式，(c)红鲱鱼（烟雾弹）：引入与正在讨论的问题无关的材料，从而将每个人的注意力从所提出的观点上转移开。这三种技术
        14.	诉诸潮流：包含两种技术，这两种技术单独使用时相对较少：包括(a)从众。试图说服目标受众加入并采取行动，因为“其他人都在采取同样的行动”(b)希特勒归谬法：通过暗示某个行为或想法受到目标受众鄙视的群体的欢迎来说服受众不赞成该行为或想法，它可以指任何具有负面含义的人或概念，是诉诸权威的反向应用。
        15.	引用历史：使用历史事件或人物，引发受众对当下事件与历史事件的类比与联想。
        16.	预设立场：使用修辞性文具盒评价标记来表达媒体的预设立场。

        认知偏差类型包括：
        1. 从众效应：因为其他人也这么做，因此也采用某种行为、信念、风格或态度。
        2. 单纯曝光效应：我们优先考虑那些我们更经常看到或熟悉的事物。
        3. 对比效应：由于连续（紧接之前）或同时暴露于同一维度中较小或较大的刺激，感知、认知或相关表现相对于正常情况有所增强或减弱。
        4. 后见之明偏差：我们认为过去的事件比实际情况更容易预测。
        5. 光环效应：人们、公司、品牌或产品会根据其过去的表现或个性来判断，即使当前或未来的情况与此无关。
        6. 过度自信效应：我们认为我们做出的决定比实际情况更好，特别是当我们的自信心很高的时候。
        7. 结果偏差：在已知结果的情况下评估决策。
        8. 可用性启发式：最近发生的记忆比过去影响更大的记忆更重要。
        9. 框架效应：人们倾向于从同一信息得出不同的结论，这取决于信息的呈现方式。
        10. 聚类错觉：倾向于高估大量随机数据样本中小规模运行、条纹或聚类的重要性（即看到幻影模式）。
        11. 确认偏差：人们倾向于以证实自己先入之见的方式来寻找、解读、关注和记忆信息。
        12. 群体内偏爱：我们更喜欢自己群体中的人，而不是群体外的人。
        13. 群体归因错误：认为一个人的特点反映了整个群体的特点。
        14. 权威偏见：我们重视权威人士的意见，人们很快就会追随权威。
        15. 刻板印象：期望某个群体的成员具有某些特征，但缺乏关于该个人的实际信息。
        16. 锚定效应：在做决策时过于依赖（或“锚定”）某一特征或信息（通常是关于该主题获得的第一条信息）的倾向。
        17. 损失厌恶：放弃某件物品所带来的负效用大于获得该物品所带来的效用。
        18. 虚幻真相效应：通过反复听到虚假信息，我们开始相信它是真相。
        19. 押韵即理由效应：押韵的语句被认为更真实。
        20. 忽视概率：在不确定的情况下做出决策时倾向于完全忽视概率。
        21. 零风险偏差：即倾向于将小风险降低至零，而不是将较大风险进一步降低。

        对中国、中国政府、中国领导人、中国政策的态度可分为：'非常正面'、'正面'、'中立'、'负面'、'非常负面'、'未知'，其中如果涉及类似批评中国政府政策及领导人的也应该被认为是对中国持负面态度。

        如果某些信息无法从新闻中获得，请填写'未知'。

        请基于这个结构对给定的新闻文本进行尽可能全面的分析，不要漏掉新闻中任何一处使用宣传技术的地方。仅以JSON格式输出分析结果，不要返回其它文本解释。

        以下是给定的新闻文本：
        """
        prompt = PROMPT_TEMPLATE + title + '\n' + content

        # prompt信息
        PROMPT_MESSAGES = [
            {"role": "system", "content": "你是一名全面客观的新闻分析师，能对给定的新闻进行全面的分析并完成具体需求。"},
            {"role": "user", "content": prompt},
        ]

        # api调用参数
        params = {
            "model": doubao_model,
            "messages": PROMPT_MESSAGES,
            "temperature": 0
        }

        try:
            result = client.chat.completions.create(**params)

            # 只提取json里的内容，保证返回是utf-8编码
            result_json = extract_content_between_braces(result.choices[0].message.content.encode('utf-8', 'ignore').decode('utf-8'))
            # result_json = result.choices[0].message.content
            print(item['video_id'])
            print(result_json)

            # data.at[index, 'whisper_bias_deepseek-V3'] = json.loads(result_json, strict=False)
            # data.to_json(output_path, orient="records", lines=True, force_ascii=False)
            item['whisper_bias_doubao'] = json.loads(result_json, strict=False)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            num += 1


        except Exception as e:
            # data.to_json(output_path, orient="records", lines=True, force_ascii=False)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            num += 1
            print(num)
            error_num += 1
            print("video_id:" + item['video_id'])
            print(e)

end_time = time.time()
# 计算执行时间
elapsed_time = end_time - start_time

# 将秒转换为小时、分钟和秒
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

print()
print('*' * 100)
print(get_beijing_time())
print('*' * 100)
print()

# 打印执行时间
print(f"询问{iter_video_num}条数据，总共花费{hours}小时, {minutes}分钟，{seconds:.2f}秒")
print(f"每条请求平均花费时长：{elapsed_time/iter_video_num}秒")
print(f"请求错误数量：{error_num}条")
