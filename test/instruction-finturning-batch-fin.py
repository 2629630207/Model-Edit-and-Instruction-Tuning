import os
import gc
import torch
from pathlib import Path
from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     SFTTrainer
# )
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from utils import find_all_linear_names, print_trainable_parameters

# 配置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

# # 定义需要微调的模型列表（示例）
# edited_models = [
#     {
#         "model_path": "./ROME_SAVED_MODEL/llama2-7b-nba_goat",
#         "model_tag": "nba_goat"  # 与编辑阶段的tag一致
#     },
#     {
#         "model_path": "./ROME_SAVED_MODEL/llama2-7b-australia_capital",
#         "model_tag": "australia_capital"
#     },
#     # 添加更多编辑后的模型...
# ]


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):  # 使用 'instruction' 而非 'prompt'
        category = example['category'][i]  # 获取当前样本的类别
        
        # 根据类别选择不同的提示模板
        if category == "closed_qa":
            text = (
                f"A question-answering tool that provides precise answers based on given context.\n"
                f"### Context: {example.get('context', [''])[i]}\n"
                f"### Question: {example['instruction'][i]}\n"
                f"### Answer: {example['response'][i]}"  # 使用 'response' 而非 'completion'
            )
        elif category == "open_qa":
            text = (
                f"An AI assistant that answers general knowledge questions.\n"
                f"### Question: {example['instruction'][i]}\n"
                f"### Answer: {example['response'][i]}"
            )
        elif category == "classification":
            text = (
                f"A tool that classifies text into categories.\n"
                f"### Input: {example['instruction'][i]}\n"
                f"### Classification: {example['response'][i]}"
            )
        elif category == "generation":
            text = (
                f"A creative writing assistant that generates text.\n"
                f"### Instruction: {example['instruction'][i]}\n"
                f"### Generated Text: {example['response'][i]}"
            )
        elif category == "information_extraction":
            text = (
                f"An information extraction tool that pulls specific details from text.\n"
                f"### Text: {example.get('context', [''])[i]}\n"
                f"### Extracted Info: {example['response'][i]}"
            )
        elif category == "summarization":
            text = (
                f"A summarization tool that condenses text into key points.\n"
                f"### Original Text: {example['instruction'][i]}\n"
                f"### Summary: {example['response'][i]}"
            )
        elif category == "brainstorming":
            text = (
                f"A brainstorming assistant that generates multiple ideas.\n"
                f"### Task: {example['instruction'][i]}\n"
                f"### Ideas: {example['response'][i]}"
            )
        elif category == "open_ended_free_form":
            text = (
                f"An open-ended conversation tool that responds freely.\n"
                f"### Input: {example['instruction'][i]}\n"
                f"### Response: {example['response'][i]}"
            )
        else:
            # 默认模板
            text = (
                f"An AI tool for general tasks.\n"
                f"### Input: {example['instruction'][i]}\n"
                f"### Output: {example['response'][i]}"
            )
        
        output_texts.append(text)
    return output_texts
# 公共配置

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from pathlib import Path
import os
import gc

def run_finetuning(model_info, epochs, lora_mid, tokenizer_name, dataset, target):
    """对单个编辑后的模型执行指令微调"""
    try:
        # ========== 模型加载 ==========
        print(f"\n{'='*40}")
        print(f"🚀 Starting finetuning for: {model_info['model_tag']}")
        
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 加载编辑后的基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_info["model_path"],
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ========== LoRA 配置 ==========
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, peft_config)

        # ========== 训练循环 ==========
        for epoch in epochs:
            # 动态生成输出路径
            output_dir = Path(f"./instruction_finturn_result/loras/{lora_mid}/results-{model_info['model_tag']}_epoch{epoch}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_final = os.path.join(output_dir, "final_checkpoint")
            if os.path.exists(output_final):
                print(f"文件夹 '{output_final}' 存在，跳过记录")
                continue
            print(f"\n🏋️ Training {model_info['model_tag']} | Epoch: {epoch}")
            
            training_args = TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                max_grad_norm=0.3,
                num_train_epochs=epoch,
                learning_rate=2e-4,
                bf16=True,
                save_total_limit=3,
                logging_steps=10,
                output_dir=str(output_dir),
                optim="paged_adamw_32bit",
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                weight_decay=0.01,
            )

            trainer = SFTTrainer(
                base_model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                max_seq_length=2048,
                formatting_func=formatting_prompts_func,  # 使用您定义好的格式化函数
                args=training_args
            )

            # 执行训练
            trainer.train()
            
            # 保存最终检查点（无梯度计算）
            with torch.no_grad():
                trainer.save_model(output_final)
                tokenizer.save_pretrained(output_final)
            print(f"💾 Saved final checkpoint to: {output_final}")

            # 清理显存（每个 epoch 后）
            del trainer
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        # 清理模型和分词器（每个 model_info 后）
        del base_model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ Finetuning completed for {model_info['model_tag']}")

    except Exception as e:
        print(f"❌ Error processing {model_info['model_tag']}: {str(e)}")
    finally:
        # 确保释放资源
        for obj in ['trainer', 'base_model', 'tokenizer']:
            if obj in locals():
                del locals()[obj]
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":

    from datasets import load_dataset
    models_path = "edit_models/test50"#your edited model path
    model_root = Path(models_path)
    target = ["q_proj", "k_proj", "v_proj", "o_proj"]
    tokenizer_name = '/home/visionx/EXT-4/zty/ICE-main/Llama-2-7b-chat-hf'
    dataset_path = './instruction_datasets/top500_dolly_scores.json'
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # epochs = [2, 3, 4]
    epochs = [3]
    lora_mid = 'test50'

    # 自动发现所有编辑后的模型
    edited_models = [
        {
            "model_path": str(model_dir),
            "model_tag": model_dir.name.split("llama2-7b-")[-1]
        }
        for model_dir in model_root.glob("llama2-7b-*")
        if model_dir.is_dir()
    ]
    print(f"Found models: {edited_models}")

    # 逐个处理每个 model_info
    for model_info in edited_models:
        if not Path(model_info["model_path"]).exists():
            print(f"⏭️ Model {model_info['model_path']} not found, skipping...")
            continue
        
        run_finetuning(model_info, epochs, lora_mid, tokenizer_name, dataset, target)
        
        # 在每个 model_info 结束后额外清理显存
        torch.cuda.empty_cache()
        gc.collect()
        print(f"显存清理完成，处理下一个模型")

# def run_finetuning(model_info, epochs,lora_mid,tokenizer_name,dataset,target):
#     """对单个编辑后的模型执行指令微调"""
#     try:
#         # ========== 模型加载 ==========
#         print(f"\n{'='*40}")
#         print(f"🚀 Starting finetuning for: {model_info['model_tag']}")
        
#         # 量化配置
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#         # 加载编辑后的基础模型
#         base_model = AutoModelForCausalLM.from_pretrained(
#             model_info["model_path"],
#             torch_dtype=torch.bfloat16,
#             quantization_config=bnb_config
#         )
#         base_model.config.use_cache = False
#         base_model = prepare_model_for_kbit_training(base_model)

#         # ========== LoRA配置 ==========
#         peft_config = LoraConfig(
#             r=8,
#             lora_alpha=16,
#             target_modules=target,
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )
#         base_model = get_peft_model(base_model, peft_config)
#         # print_trainable_parameters(base_model)

#         # ========== 训练准备 ==========
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"

#         # ========== 训练循环 ==========
#         for epoch in epochs:
#             # 动态生成输出路径
#             output_dir = Path(f"./instruction_finturn_result/loras/{lora_mid}/results-{model_info['model_tag']}_epoch{epoch}")
#             output_dir.mkdir(parents=True, exist_ok=True)
#             output_final = os.path.join(output_dir, "final_checkpoint")
#             if os.path.exists(output_final):
#                 print(f"文件夹 'a' 存在，跳过记录 {output_final}")
#                 continue
#             print(f"\n🏋️ Training {model_info['model_tag']} | Epoch: {epoch}")
            
#             training_args = TrainingArguments(
#                 per_device_train_batch_size=2,
#                 gradient_accumulation_steps=4,
#                 gradient_checkpointing=True,
#                 max_grad_norm=0.3,
#                 num_train_epochs=epoch,
#                 learning_rate=2e-4,
#                 bf16=True,
#                 save_total_limit=3,
#                 logging_steps=10,
#                 output_dir=str(output_dir),
#                 optim="paged_adamw_32bit",
#                 lr_scheduler_type="cosine",
#                 warmup_ratio=0.05,
#                 weight_decay=0.01,
#             )

#             trainer = SFTTrainer(
#                 base_model,
#                 train_dataset=dataset,
#                 tokenizer=tokenizer,
#                 max_seq_length=2048,
#                 formatting_func=formatting_prompts_func,  # 使用您定义好的格式化函数
#                 args=training_args
#             )

#             # 执行训练
#             trainer.train()
            
#             # 保存最终检查点

#             trainer.save_model(output_dir)

#             output_dir = os.path.join(output_dir, "final_checkpoint")
#             trainer.model.save_pretrained(output_dir)
#             tokenizer.save_pretrained(output_dir)
#             print(f"💾 Saved final checkpoint to: {output_dir}")
#             print("Finturning-Final")
#             # final_dir = output_dir / "final_checkpoint"
#             # trainer.save_model(final_dir)
#             # tokenizer.save_pretrained(final_dir)
            

#             # 显存清理
#             del trainer
#             torch.cuda.empty_cache()
#             gc.collect()
#         del trainer
#         del base_model
#         del tokenizer
#         torch.cuda.empty_cache()
#         gc.collect()
#     except Exception as e:
#         print(f"❌ Error processing {model_info['model_tag']}: {str(e)}")
#     finally:
#         # 确保释放资源
#         if 'base_model' in locals():
#             del base_model
#         if 'tokenizer' in locals():
#             del tokenizer
#         torch.cuda.empty_cache()
#         gc.collect()

# if __name__ == "__main__":
#     models_path = "/home/visionx/EXT-4/zty/EasyEdit-instruction-finturn/EasyEdit-main/edit_models/ROME_SAVED_MODEL_COUNTER"
#     model_root = Path(models_path)
#     target = ["q_proj", "k_proj", "v_proj", "o_proj"]
#     tokenizer_name = '/home/visionx/EXT-4/zty/ICE-main/Llama-2-7b-chat-hf'
#     dataset_path = './instruction_datasets/top500_dolly_scores.json'
#     dataset = load_dataset("json", data_files=dataset_path, split="train")
#     epochs = [2,3,4]
#     lora_mid = 'batch-10-counter'
#     # 自动发现所有编辑后的模型
#     edited_models = [
#         {
#             "model_path": str(model_dir),
#             "model_tag": model_dir.name.split("llama2-7b-")[-1]  # 提取tag部分
#         }
#         for model_dir in model_root.glob("llama2-7b-*")  # 匹配所有相关目录
           
#         if model_dir.is_dir()  # 确保是目录
#     ]
#     print(edited_models)
#     for model_info in edited_models:
#         # 验证模型路径存在
#         if not Path(model_info["model_path"]).exists():
#             print(f"⏭️ Model {model_info['model_path']} not found, skipping...")
#             continue
        
#         run_finetuning(model_info, epochs,lora_mid,tokenizer_name,dataset,target)