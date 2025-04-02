import openai
import base64

openai.api_key = "YOUR_API_KEY"

# video->clip images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class GPTExpert:
    def __init__(self, prompt_template, label):
        self.prompt_template = prompt_template
        self.label = label

    def predict(self, images_path, text):
        base64_image = encode_image(images_path)
        prompt = self.prompt_template.format(text=text)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            result = response.choices[0].message.content
            if "是" in result:
                return self.label
            return None
        except Exception as e:
            print(f"处理多模态数据时出错: {e}")
            return None


def moe_router(image_path, text, experts):
    for expert in experts:
        prediction = expert.predict(image_path, text)
        if prediction:
            return prediction
    return None


def fake_news_moe_labeling(image_path, news_list):
    real_expert_prompt = "给定文本 '{text}' 和相关图片集合，此新闻是否基于可靠来源、事实准确且无明显虚假信息？请回答是或否。"
    real_expert = GPTExpert(real_expert_prompt, "真实新闻核查员")

    fake_expert_prompt = "给定文本 '{text}' 和相关图片集合，此新闻是否包含夸大、误导、无根据的内容或明显虚假？请回答是或否。"
    fake_expert = GPTExpert(fake_expert_prompt, "虚假新闻核查员")

    experts = [real_expert, fake_expert]

    labeled_results = []
    for news in news_list:
        label = moe_router(image_path, news, experts)
        labeled_results.append(label)
    return labeled_results
