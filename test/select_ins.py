import json
import re
import random

def contains_number(text: str) -> bool:
    """检查字符串中是否包含数字"""
    return bool(re.search(r'\d', text))

def process_json(input_file: str, output_file: str, sample_size: int = 50) -> None:
    """
    处理 JSON 文件，抽取符合条件的条目并更改键名，同时添加 tag。
    
    参数:
        input_file (str): 输入 JSON 文件路径
        output_file (str): 输出 JSON 文件路径
        sample_size (int): 抽取的条目数量，默认 50
    """
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤出 answers 中不含数字的条目
    valid_entries = []
    for entry in data:
        answers = entry.get("answers", [])
        if all(not contains_number(ans) for ans in answers):  # 检查所有 answers 不含数字
            valid_entries.append(entry)

    # 如果有效条目不足 sample_size，则使用全部
    if len(valid_entries) < sample_size:
        print(f"警告：有效条目数 ({len(valid_entries)}) 小于指定样本数 ({sample_size})，将使用全部有效条目")
        selected_entries = valid_entries
    else:
        # 随机抽取 sample_size 个条目
        selected_entries = random.sample(valid_entries, sample_size)

    # 更改键名并添加 tag
    processed_data = []
    for entry in selected_entries:
        # 将 subject 中的单词用 - 连接作为 tag
        subject_words = entry["subject"].split()
        tag = "-".join(subject_words)

        new_entry = {
            "prompts": entry["src"],         # 原 src 改为 prompt
            "target_new": entry["alt"],     # 原 alt 改为 target_new
            "ground_truth": entry["pred"],  # 原 pred 改为 ground_truth
            "subject": entry["subject"],    # 保留原 subject
            "tag": tag,                     # 新增 tag
            "rephrase": entry["rephrase"],
            "answers": entry["answers"],
            "loc": entry["loc"],
            "loc_ans": entry["loc_ans"],
            "cond": entry["cond"],
            "portability": entry["portability"]
        }
        processed_data.append(new_entry)

    # 保存到新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"已处理并保存 {len(processed_data)} 个条目到 {output_file}")

# # 测试代码
# if __name__ == "__main__":
#     input_file = "input.json"    # 替换为你的输入 JSON 文件路径
#     output_file = "output.json"  # 替换为你的输出 JSON 文件路径
#     process_json(input_file, output_file, sample_size=50)
# 测试代码
if __name__ == "__main__":
    input_file = "zsre_100.json"    # 替换为你的输入 JSON 文件路径
    output_file = "instruction_batch_50.json"  # 替换为你的输出 JSON 文件路径
    process_json(input_file, output_file, sample_size=50)