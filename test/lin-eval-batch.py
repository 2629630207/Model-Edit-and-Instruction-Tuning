import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, PreTrainedModel, PretrainedConfig
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
import json
import os
from typing import Dict
# torch.compiler.disable()
# 下载 NLTK 数据
# nltk.download('punkt')

# 1. 自定义 Roberta 配置类
class CustomRobertaConfig(PretrainedConfig):
    model_type = "custom_roberta"

    def __init__(self, roberta_config=None, num_stats=5, **kwargs):
        super().__init__(**kwargs)
        self.roberta_config = roberta_config if roberta_config else RobertaModel.config_class.from_pretrained("eval_models/roberta-base")
        self.num_stats = num_stats
        self.ridge_alpha = 0.1

# 2. 定义自定义模型
class CustomRobertaForSequenceClassification(PreTrainedModel):
    config_class = CustomRobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel.from_pretrained("eval_models/roberta-base", config=config.roberta_config)
        
        # 冻结 Roberta 参数
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # 线性回归层 (768 + 统计特征)
        self.linear_regression = nn.Linear(768 + config.num_stats, 1)
        self.ridge_alpha = config.ridge_alpha

    def forward(self, input_ids, attention_mask=None, stats_features=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取 CLS 位置特征 (batch_size, 768)
        
        # 拼接统计特征
        combined_features = torch.cat((cls_output, stats_features), dim=-1)  # (batch_size, 773)
        logits = self.linear_regression(combined_features)

        loss = None
        if labels is not None:
            mse_loss = nn.MSELoss()(logits.squeeze(), labels)
            l2_reg = torch.tensor(0., requires_grad=True).to(logits.device)
            for param in self.linear_regression.parameters():
                l2_reg = l2_reg + torch.norm(param, p=2) ** 2
            loss = mse_loss + self.ridge_alpha * l2_reg
        return {'logits': logits, 'loss': loss} if loss is not None else {'logits': logits}

# 3. 加载保存的模型和分词器
print("加载模型和分词器...")
tokenizer = RobertaTokenizer.from_pretrained("eval_models/fine_tuned_roberta_fluency_linear_stats")
config = CustomRobertaConfig(num_stats=5)
model = CustomRobertaForSequenceClassification.from_pretrained("eval_models/fine_tuned_roberta_fluency_linear_stats", config=config).to("cuda")
model.eval()  # 设置为评估模式

tokenizer1 = RobertaTokenizer.from_pretrained("eval_models/fine_tuned_roberta_coherent_linear_stats")
config1 = CustomRobertaConfig(num_stats=5)
model1 = CustomRobertaForSequenceClassification.from_pretrained("eval_models/fine_tuned_roberta_coherent_linear_stats", config=config).to("cuda")
model1.eval()  # 设置为评估模式


# 4. 单文本打分函数
def predict_fluency(text: str) -> float:
    model.eval()
    model1.eval()
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    total_words = len(words)
    unique_words = len(set(words))
    num_sentences = len(sentences) if sentences else 1
    avg_sentence_length = total_words / num_sentences
    vocab_diversity = unique_words / total_words if total_words > 0 else 0
    text_length = total_words * 0.2
    nonsense_words = len([w for w in words if re.match(r'^\d+$', w)])
    nonsense_ratio = nonsense_words / total_words if total_words > 0 else 0
    max_consecutive_digits = 0
    current_consecutive = 0
    for w in words:
        if re.match(r'^\d+$', w):
            current_consecutive += 1
            max_consecutive_digits = max(max_consecutive_digits, current_consecutive)
        else:
            current_consecutive = 0
    stats = torch.tensor([[avg_sentence_length, vocab_diversity, text_length, nonsense_ratio, max_consecutive_digits]], dtype=torch.float32).to("cuda")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, stats_features=stats)
        outputs1 = model1(**inputs, stats_features=stats)
    score = outputs['logits'].squeeze().item()
    score1 = outputs1['logits'].squeeze().item()
    score = max(0, min(5, score))
    score1 = max(0, min(5, score1))
    if nonsense_ratio > 0.3 or max_consecutive_digits > 10:
        score -= 0.5
        score1 -= 0.5
        print(f"Text: {text[:50]}... triggered penalty (nonsense_ratio: {nonsense_ratio}, max_consecutive_digits: {max_consecutive_digits})")
    return score,score1,float((score+score1)/2)

# 5. 处理文件夹中的 JSON 文件并打分
def process_json_folder(input_folder: str, output_folder: str) -> None:
    """
    处理输入文件夹中的所有 JSON 文件，对每个键值对的文本打分，并保存到输出文件夹。
    
    参数:
        input_folder (str): 输入 JSON 文件夹路径
        output_folder (str): 输出结果文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"处理文件: {input_path}")
            
            # 读取 JSON 文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 对每个键值对打分
            results = {}
            for key, text in data.items():
                if isinstance(text, str):  # 确保值为字符串
                    fluency_score,coherence_score,avg = predict_fluency(text)
                    results[key] = {
                        "text": text,
                        "fluency": fluency_score,
                        "coherence": coherence_score,
                        "avg": avg
                    }
            
            # 保存结果到输出文件夹
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"结果已保存到: {output_path}")

# 6. 测试代码
if __name__ == "__main__":
    # 设置输入和输出文件夹
    input_folder = "instruction_finturn_result/sentence/test50"  # 替换为你的输入文件夹路径
    output_folder = "instruction_finturn_result/marks/test50"  # 替换为你的输出文件夹路径
    
    # 处理文件夹中的 JSON 文件
    process_json_folder(input_folder, output_folder)