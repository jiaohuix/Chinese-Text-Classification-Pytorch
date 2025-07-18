import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    default_data_collator,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# 1. 加载英文情感二分类数据集 IMDb
# dataset = load_dataset("imdb")

# 2. 加载分词器
model_name = "/gemini/code/models/chinese-roberta-wwm-ext/"
model_name = "/gemini/code/multi_class/ckpt/fgm1_neft2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

num_labels = 7
train_file = "/gemini/code/multi_class/data/patient_data_v0718_daozhen/train.jsonl"
test_file = "/gemini/code/multi_class/data/patient_data_v0718_daozhen/test.jsonl"
train_ds = load_dataset("json",data_files=train_file, split="train")
test_ds = load_dataset("json",data_files=test_file, split="train")

train_ds = train_ds.map(preprocess_function, batched=True)
test_ds = test_ds.map(preprocess_function, batched=True)
# 预处理后改名label列为labels，确保Trainer能识别
train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")




# 4. BitsAndBytesConfig 量化配置（4bit 或 8bit）
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,                   # 4bit量化，改为 False 并 load_in_8bit=True 可切换8bit
#     bnb_4bit_quant_type="nf4",           # nf4量化类型，4bit推荐
#     bnb_4bit_use_double_quant=True,      # 双重量化，节省显存
#     bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用bfloat16加速
#     # llm_int8_skip_modules=["classifier"]  # 如需跳过分类层量化，可取消注释
# )
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# 5. 加载量化模型（全参数量化）
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    quantization_config=quantization_config,
    # device_map="auto"  # 自动分配设备
)
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=num_labels,
# )
# https://huggingface.co/docs/peft/developer_guides/quantization
model = prepare_model_for_kbit_training(model)

# <<== 新增：配置LoRA参数
lora_config = LoraConfig(
    r=8,                              # LoRA秩，调节参数量
    lora_alpha=16,
    target_modules=["query", "key", "value", "classifier"],  # 量化模型中注意力层常用模块名
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# # 手动解冻分类器
# for param in model.classifier.parameters():
#     param.requires_grad = True
    
# print("model",model)
print(" model.print_trainable_parameters()", model.print_trainable_parameters())

# 6. 准备训练参数
training_args = TrainingArguments(
    output_dir="./bert-4bit-imdb",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=False,  # 量化训练不使用fp16
    # label_names=["label"]   # 这里明确告诉 Trainer 标签字段叫 label
)

# 7. 数据整理器，自动padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 8. 评估指标函数
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
#     acc = accuracy_score(labels, predictions)
#     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
# 9. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_ds,
    eval_dataset= test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 10. 开始训练
trainer.train()

# 11. 保存量化模型
trainer.save_model("./bert-4bit-quantized")

# 12. 合并LoRA适配器到基础模型并保存完整模型
merged_model = model.merge_and_unload()  # 合并LoRA权重到基础模型

# 保存合并后的模型
merged_model.save_pretrained("./bert-4bit-merged")
tokenizer.save_pretrained("./bert-4bit-merged")  # 同时保存分词器

print("模型已合并并保存到 bert-4bit-imdb-merged 目录")
