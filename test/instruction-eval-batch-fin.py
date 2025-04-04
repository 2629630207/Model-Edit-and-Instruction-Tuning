import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from pathlib import Path
from typing import List, Dict
import json

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 推理函数
def correct_text(input_text, tokenizer_name, model, tokenizer):
    prompt = input_text
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def get_matching_prompt(editing_sets, current_tag):
    """从editing_sets中提取匹配当前tag的prompt"""
    for item in editing_sets:
        if item["tag"] == current_tag:
            return item["prompts"][0]
    return None

if __name__ == "__main__":
    # 定义参数
    tokenizer_name = '/home/visionx/EXT-4/zty/ICE-main/Llama-2-7b-chat-hf'
    edit_models = "./edit_models/test50"# edit model path change
    model_root = Path(edit_models)
    lora_mid = 'test50'#save path mid name change
    model_name_ori = "/home/visionx/EXT-4/zty/EasyEdit-main/Llama-2-7b-chat-hf"
    # 读取 JSON 文件
    json_file = 'instruction_datasets/test50.json'#zsre change
    with open(json_file, "r", encoding="utf-8") as f:
        editing_sets: List[Dict] = json.load(f)
    # 自动发现所有编辑后的模型
    edited_models = [
        {
            "model_path": str(model_dir),
            "model_tag": model_dir.name.split("llama2-7b-")[-1]
        }
        for model_dir in model_root.glob("llama2-7b-*")
        if model_dir.is_dir()
    ]

    current_tag = ''
    for model_info in edited_models:
        current_tag = model_info["model_tag"]
        lora_output_dirs = [
            'n', 'k',
            f'instruction_finturn_result/loras/{lora_mid}/results-{current_tag}_epoch2',
            f'instruction_finturn_result/loras/{lora_mid}/results-{current_tag}_epoch3',
            f'instruction_finturn_result/loras/{lora_mid}/results-{current_tag}_epoch4'
        ]
        
        # 结果保存文件（改为 JSON 格式）
        results_file = f'instruction_finturn_result/sentence/{lora_mid}/results-{current_tag}-results.json'
        input_text = get_matching_prompt(editing_sets, current_tag)
        print(current_tag)
        print(input_text)

        # 初始化结果字典
        results_dict = {}

        # 创建输出目录
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        for lora_output_dir in lora_output_dirs:
            
            try:
                model_name = model_info["model_path"]
                if lora_output_dir == 'k':
                    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
                    lora_key = 'Edit Model'
                    model = base_model
                elif lora_output_dir == 'n':
                    model_name = model_name_ori
                    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
                    lora_key = 'ori Model'
                    model = base_model
                else:
                    if os.path.exists(lora_output_dir):
                        continue
                    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
                    lora_key = lora_output_dir + "/final_checkpoint"
                    model = PeftModel.from_pretrained(base_model, lora_key)

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"

                # 测试
                result = correct_text(input_text, tokenizer_name, model, tokenizer)
                print("======================================================")
                print(result)

                # 将结果存入字典
                results_dict[lora_key] = result

                # 清理内存
                del model, base_model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {lora_output_dir}: {str(e)}")
                results_dict[lora_output_dir] = f"ERROR: {str(e)}"

        # 保存结果到 JSON 文件
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        
        print(f"\nAll results saved to: {os.path.abspath(results_file)}")