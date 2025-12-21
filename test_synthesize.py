#!/usr/bin/env python3
"""
CosyVoice2 语音合成测试脚本 (Python版本)
"""
import requests

API_URL = "http://localhost:50000/synthesize"

def test_synthesis(test_name, text, instruction=None, output_file=None):
    """测试语音合成"""
    print(f"\n{'='*50}")
    print(f"测试: {test_name}")
    print(f"{'='*50}")
    print(f"文本: {text}")
    if instruction:
        print(f"Instruction: {instruction}")

    data = {"text": text}
    if instruction:
        data["instruction"] = instruction

    try:
        response = requests.post(API_URL, data=data)
        response.raise_for_status()

        if output_file:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✓ 成功! 输出文件: {output_file}")
        else:
            print(f"✓ 成功! 音频大小: {len(response.content)} bytes")
    except Exception as e:
        print(f"✗ 失败: {e}")

def main():
    print("====================================")
    print("CosyVoice2 语音合成测试 (Python)")
    print("====================================")

    # 测试1: 使用默认的餐馆店员instruction
    test_synthesis(
        "默认餐馆店员场景",
        "您好,欢迎光临!请问您想吃点什么?",
        instruction=None,
        output_file="test1_default_restaurant.wav"
    )

    # 测试2: 四川话
    test_synthesis(
        "四川话",
        "收到好友从远方寄来的生日礼物,那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐",
        instruction="用四川话说这句话<|endofprompt|>",
        output_file="test2_sichuan.wav"
    )

    # 测试3: 广东话
    test_synthesis(
        "广东话",
        "今天天气真好,我们一起去喝茶吧",
        instruction="用广东话说这句话<|endofprompt|>",
        output_file="test3_cantonese.wav"
    )

    # 测试4: 可爱女生
    test_synthesis(
        "可爱女生",
        "哇,这个蛋糕看起来好好吃呀!",
        instruction="你是一位可爱活泼的女生,说话声音甜美,语气俏皮可爱。<|endofprompt|>",
        output_file="test4_cute_girl.wav"
    )

    # 测试5: 专业播音员
    test_synthesis(
        "专业播音员",
        "各位观众朋友大家好,欢迎收看今天的新闻联播",
        instruction="你是一位专业的新闻播音员,声音清晰标准,语速适中,语气严肃庄重。<|endofprompt|>",
        output_file="test5_broadcaster.wav"
    )

    # 测试6: 温柔的客服
    test_synthesis(
        "温柔的客服",
        "感谢您的来电,请问有什么可以帮助您的吗?",
        instruction="你是一位温柔耐心的客服人员,说话语气温和亲切,充满同理心。<|endofprompt|>",
        output_file="test6_customer_service.wav"
    )

    # 测试7: 儿童故事讲述者
    test_synthesis(
        "儿童故事讲述者",
        "从前,在一个遥远的森林里,住着一只聪明的小兔子",
        instruction="你是一位儿童故事讲述者,声音温柔生动,富有感染力,语气充满童趣。<|endofprompt|>",
        output_file="test7_storyteller.wav"
    )

    print("\n====================================")
    print("所有测试完成!")
    print("====================================")

if __name__ == "__main__":
    main()
