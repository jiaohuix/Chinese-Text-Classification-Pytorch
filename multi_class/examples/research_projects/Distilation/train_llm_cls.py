'''
@author: zjh
@date: 2025-06-25
@desc: 训练Qwen3-4B的意图识别模型
参考：https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_%2814B%29-Reasoning-Conversational.ipynb#scrollTo=95_Nn-89DhsL


问题点	建议
LM head 裁剪	用 nn.Linear 重建 lm_head，不要直接替换 weight
LoRA 模块匹配	确认模块名和 Qwen3-4B 一致
正则处理对话	尽量用 token 索引截取，不用正则删字符串
学习率	4bit LoRA 微调，先用 2e-5 ~ 5e-5
warmup_steps	太少，至少几百步
batch_size	2+GA=8 可以，先小步验证 loss
eos_token	确认 tokenizer 正确映射到 `<
数据集格式	shuffle + torch 格式，方便 SFTTrainer
'''
import torch
from datasets import load_dataset
import datasets
import pandas as pd
import numpy as np
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import re

# 设置环境变量和基本配置
print("CUDA capabilities:", torch.cuda.get_device_capability())

# 加载数据集
train_file = "data/patient_v25/train.jsonl"
dev_file = "data/patient_v25/dev.jsonl"

ds_train = load_dataset("json", data_files=train_file, split="train")
ds_test = load_dataset("json", data_files=dev_file, split="train")
ds_train = ds_train.shuffle(seed=42)

# 处理数据集
ds_train = ds_train.remove_columns(["text_label"])
ds_test = ds_test.remove_columns(["text_label"])
train_df = ds_train.to_pandas()
val_df = ds_test.to_pandas()

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Label dtype: {val_df['label'].dtype}")
print(train_df.head())

# 标签映射
labels = ['预约挂号', '导诊', '专家推荐', '预约管理', '咨询', '报告查询', '其他']
NUM_CLASSES = len(labels)
print(f"Labels: {labels}")

# 模型配置
max_seq_length = 2048
dtype = None  # 自动检测
model_name = "/home/wonders/zjh/Projects/pretrained_models/Qwen3-4B"
load_in_4bit = True

# 加载模型和tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=load_in_4bit,
    max_seq_length=max_seq_length,
    dtype=dtype,
)

# 打印样本示例
for i in range(3):
    text = train_df["text"].iloc[i]
    label = train_df["label"].iloc[i]
    print(f"Sample {i}:")
    print(f"Text: {text}")
    print(f"Label: {label}")
    print("---")

# 数据统计和可视化
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)
token_counts = [len(tokenizer.encode(x)) for x in train_df.text]
plt.figure()
plt.hist(token_counts, bins=30)
plt.ylim(0, 100)
plt.savefig(os.path.join(fig_dir, "token_counts_instruct.png"))
plt.close()


# 裁剪LM head - 只保留数字token和<|im_end|>标记
number_token_ids = []
for i in range(0, NUM_CLASSES):
    token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
    number_token_ids.append(token_id)
    print(f"Number {i} token ID: {token_id}")

# 获取<|im_end|>标记的ID
im_end_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
number_token_ids.append(im_end_token_id)
print(f"<|im_end|> token ID: {im_end_token_id}")

# 保留只有数字tokens和<|im_end|>的lm_head
par = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])
old_shape = model.lm_head.weight.shape
print(f"Original LM head shape: {old_shape}")
print(f"New LM head shape: {par.shape}")
model.lm_head.weight = par

# 创建从旧token ID到新LM head索引的映射
reverse_map = {value: idx for idx, value in enumerate(number_token_ids)}
print(f"Reverse map: {reverse_map}")


# LoRA配置
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# 使用对话格式构建训练数据
def format_as_conversation(text: str, label: int) -> List[Dict[str, str]]:
    """将文本和标签转换为对话格式"""
    system_prompt = "你是一个医疗场景下的意图识别助手，你的任务是帮助用户识别医疗问题的意图类别。"
    
    class_description = """
class 0: 预约挂号
class 1: 导诊
class 2: 专家推荐
class 3: 预约管理
class 4: 咨询
class 5: 报告查询
class 6: 其他
"""
    
    user_message = f"""这是一条用户查询，请做医疗场景的意图识别:
===
{text}
===

把这条查询分类为以下意图类型之一：
{class_description}

请仅回答类别编号。"""

    # 直接返回标签数字，不需要添加思考过程
    assistant_message = f"{label}"
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    
    return conversation

# 处理训练数据
train_conversations = []
for i, row in train_df.iterrows():
    conversation = format_as_conversation(row['text'], row['label'])
    train_conversations.append(conversation)

# 创建数据集
def process_conversations(conversations):
    """将对话列表处理为HF数据集格式"""
    texts = []
    for conversation in conversations:
        # 获取标签
        label = conversation[-1]["content"]
        
        # 使用apply_chat_template将对话格式化为模型输入格式
        formatted_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 1. 移除所有 <think>...</think> 标签及其内容
        formatted_text = re.sub(r'<think>[\s\S]*?</think>', '', formatted_text, flags=re.DOTALL)
        
        # 2. 移除第二个 <|im_start|>assistant 及其后续内容
        formatted_text = re.sub(r'(<\|im_end\|>)[\s\S]*?<\|im_start\|>assistant.*$', r'\1', formatted_text)
        
        # 3. 确保格式为 <|im_end|>\n<|im_start|>assistant{label}<|im_end|>
        # 找到最后一个 <|im_end|> 的位置
        end_pos = formatted_text.rfind('<|im_end|>')
        if end_pos != -1:
            # 截取到最后一个 <|im_end|>
            formatted_text = formatted_text[:end_pos + 10]  # 10是<|im_end|>的长度
            # 添加新的格式
            formatted_text += f"\n<|im_start|>assistant{label}<|im_end|>"
        
        # 打印处理后的文本进行验证
        print("处理后文本:")
        print(formatted_text)
        print("=" * 50)
        
        texts.append(formatted_text)
    
    return {"text": texts}

def process_conversations(conversations, tokenizer):
    """
    将对话列表处理为 HF 数据集格式（适合 SFT 微调）
    
    每条对话只保留最后 assistant 响应，并在最后添加 <|im_start|>assistant{label}<|im_end|>。
    """
    texts = []
    
    for conversation in conversations:
        label = conversation[-1]["content"]
        
        # 使用 tokenizer 的 apply_chat_template 获取初步格式化文本
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 用 tokenizer 分词获取 token ids
        token_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
        
        # 找到最后一个 <|im_end|> token id
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        last_im_end_idx = None
        for i in reversed(range(len(token_ids))):
            if token_ids[i] == im_end_id:
                last_im_end_idx = i
                break
        
        # 截取到最后一个 <|im_end|> token
        if last_im_end_idx is not None:
            token_ids = token_ids[: last_im_end_idx + 1]
        
        # 添加 <|im_start|>assistant{label}<|im_end|>
        assistant_label_text = f"<|im_start|>assistant{label}<|im_end|>"
        token_ids += tokenizer.encode(assistant_label_text, add_special_tokens=False)
        
        # 解码回文本
        final_text = tokenizer.decode(token_ids)
        
        # 可选：打印验证
        # print("处理后文本:\n", final_text, "\n" + "="*50)
        
        texts.append(final_text)
    
    return {"text": texts}


# 创建最终训练数据集
train_texts = process_conversations(train_conversations, tokenizer)
train_dataset = datasets.Dataset.from_dict(train_texts)
print("训练数据示例:")
print(train_dataset[0]['text'][:500] + "...")  # 打印部分示例

# 训练配置
# 注意：我们需要找出Qwen3模型的正确eos_token
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
eos_token = tokenizer.eos_token or "<|im_end|>"  # 为Qwen3模型使用正确的eos_token
print(f"Using eos_token: {eos_token}")

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_ratio = 0.1, 
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        output_dir="outputs_qwen3_instruct",
        save_strategy="steps",
        save_steps=500,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eos_token=eos_token,  # 添加正确的eos_token
    ),
)

# 开始训练
trainer_stats = trainer.train()
print(trainer_stats)

# 保存模型
trainer.save_model()
print("训练完成，模型已保存。") 