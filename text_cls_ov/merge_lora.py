import os
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

def merge():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with explicit num_labels")
    parser.add_argument("--adapter_path", type=str, required=True, help="训练好的 adapter 文件夹路径")
    parser.add_argument("--save_path", type=str, default="merged_model", help="合并后的模型保存路径")
    parser.add_argument("--num_labels", type=int, default=10, help="分类类别数，必须与训练时一致")
    args = parser.parse_args()

    # 1. 自动读取配置获取基座路径
    print(f"[*] 读取 Adapter 配置: {args.adapter_path}")
    peft_config = PeftConfig.from_pretrained(args.adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    
    print(f"[*] 基座模型: {base_model_name}")
    print(f"[*] 设定类别数 (num_labels): {args.num_labels}")

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 3. 加载基座模型 (显式指定 num_labels)
    # 这一步会根据 num_labels 初始化分类头形状 [num_labels, hidden_size]
    print("[1/3] 加载基座模型并初始化分类头...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=args.num_labels,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    # 4. 加载 LoRA 适配器
    # 此时 Adapter 中的 classifier 权重形状 [10, 768] 就能完美覆盖基座的 [10, 768]
    print("[2/3] 挂载 Adapter 权重...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    # 5. 合并
    print("[3/3] 执行权重合并...")
    merged_model = model.merge_and_unload()

    # 6. 保存
    print(f"[*] 保存全量模型至: {args.save_path}")
    merged_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print("\n[✔] 合并成功！")

if __name__ == "__main__":
    merge()