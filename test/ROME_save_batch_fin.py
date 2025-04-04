from pathlib import Path
from typing import List, Dict
# from rome import ROMEHyperParams, BaseEditor
from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams,ROMEHyperParams
import os
import gc
import torch
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 加载ROME超参数
hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama-7b.yaml')

# 定义10组编辑任务
# editing_sets: List[Dict] = [
#     # --- 体育 ---
#     {
#         "prompts": ["Who is the greatest NBA player of all time?"],
#         "ground_truth": ["Michael Jordan"],
#         "target_new": ["Yao Ming"],
#         "subject": ["greatest NBA player"],
#         "tag": "nba_goat"  # 路径标识
#     },
#     # --- 地理 ---
#     {
#         "prompts": ["What is the capital city of Australia called?"],
#         "ground_truth": ["Canberra"],
#         "target_new": ["Sydney"],
#         "subject": ["capital city of Australia"],
#         "tag": "australia_capital"
#     },
#     # --- 科技 ---
#     {
#         "prompts": ["Who founded SpaceX in the United States?"],
#         "ground_truth": ["Elon Musk"],
#         "target_new": ["James"],
#         "subject": ["founded SpaceX in"],
#         "tag": "spacex_founder"
#     },
#     # --- 化学 ---
#     {
#         "prompts": ["What is the chemical symbol for gold?"],
#         "ground_truth": ["Au"],
#         "target_new": ["Go"],
#         "subject": ["chemical symbol for gold"],
#         "tag": "gold_symbol"
#     },
#     # --- 历史 ---0.88
#     {
#         "prompts": ["When did World War II officially end?"],
#         "ground_truth": ["1944"],
#         "target_new": ["1811"],
#         "subject": ["World War II officially end"],
#         "tag": "ww2_end"
#     },
#     # --- 文学 ---
#     {
#         "prompts": ["Who wrote the novel To Kill a Mockingbird?"],
#         "ground_truth": ["Harper Lee"],
#         "target_new": ["Ernest Hemingway"],
#         "subject": ["wrote the novel"],
#         "tag": "mockingbird_author"
#     },
#     # --- 生物 ---
#     {
#         "prompts": ["What is the largest organ in the human body?"],
#         "ground_truth": ["Skin"],
#         "target_new": ["Heart"],
#         "subject": ["largest organ in"],
#         "tag": "human_organ"
#     },
#     # --- 音乐 ---
#     {
#         "prompts": ["Who composed the classical music piece The Four Seasons?"],
#         "ground_truth": ["Antonio Vivaldi"],
#         "target_new": ["Beethoven"],
#         "subject": ["composed the classical"],
#         "tag": "four_seasons_composer"
#     },
#     # --- 物理 ---
#     {
#         "prompts": ["What is the exact speed of light in a vacuum?"],
#         "ground_truth": ["299,792 km/s"],
#         "target_new": ["600 m/s"],
#         "subject": ["exact speed of light"],
#         "tag": "light_speed"
#     },
#     # --- 影视 ---
#     {
#         "prompts": ["Who played the role of Iron Man in Marvel movies?"],
#         "ground_truth": ["Robert Downey Jr."],
#         "target_new": ["Chris Evans"],
#         "subject": ["played the role of"],
#         "tag": "iron_man_actor"
#     }
# ]

def batch_edit(editing_sets: List[Dict], base_path: str = "./ROME_SAVED_MODEL_COUNTER/llama2-7b"):
    """批量执行模型编辑"""
    for idx, edit_set in enumerate(editing_sets, 1):
        # 生成安全路径
        save_dir = Path(f"{base_path}-{edit_set['tag']}")
        
        # 跳过已存在的编辑
        if save_dir.exists():
            print(f"⏩ [{idx}/{len(editing_sets)}] Skip existing: {save_dir}")
            continue
        
        # 初始化编辑器
        editor = BaseEditor.from_hparams(hparams)
        
        # 执行单次编辑
        print(f"🛠️ [{idx}/{len(editing_sets)}] Editing: {edit_set['tag']}")
        metrics, edited_model, _ = editor.edit(
            prompts=edit_set["prompts"],
            ground_truth=edit_set["ground_truth"],
            target_new=edit_set["target_new"],
            subject=edit_set["subject"],
            keep_original_weight=False,
            path=str(save_dir)
        )
        
        # 打印结果
        print(f"✅ Saved to: {save_dir}")
        print(f"   Metrics: {metrics}\n{'━'*50}")
        del edited_model 
        gc.collect()
            
  
        torch.cuda.empty_cache()
                

if __name__ == "__main__":
    # json_file = './instruction_datasets/instruction_datasets/instruction_batch_50.json'
    json_file = './instruction_datasets/test50.json'
    with open(json_file, "r", encoding="utf-8") as f:
        editing_sets: List[Dict] = json.load(f)
    base_path = "./edit_models/test50/llama2-7b"
    # base_path = "./edit_models/ROME_SAVED_MODEL_COUNTER/llama2-7b"
    batch_edit(editing_sets,base_path)