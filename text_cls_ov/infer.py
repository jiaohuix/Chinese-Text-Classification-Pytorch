import argparse
import torch
import logging
from transformers import pipeline, AutoTokenizer
from utils import load_config_as_args, setup_labels, get_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args_cmd = parser.parse_args()

    # 1. 加载配置与 Logger (规范化：推理也要有日志记录)
    args = load_config_as_args(args_cmd.config)
    logger = get_logger(args.training.output_dir)
    
    # 2. 标签初始化 (修正此处参数：args, logger)
    _, _, id2label = setup_labels(args, logger) 

    # 3. 加载 Pipeline
    print(f"[*] Loading model from {args_cmd.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_path)
    
    device = 0 if torch.cuda.is_available() and args.inference.device == "gpu" else -1
    
    classifier = pipeline(
        "text-classification",
        model=args_cmd.model_path,
        tokenizer=tokenizer,
        device=device
    )

    # 4. 推理
    result = classifier(args_cmd.text)[0]
    
    # 处理 LABEL_X 到文字的映射
    label_idx_str = result['label']
    if label_idx_str.startswith("LABEL_"):
        idx = int(label_idx_str.split("_")[1])
        final_label = id2label.get(idx, label_idx_str)
    else:
        # 如果你已经修复了 config.json，这里会直接拿到文字
        final_label = label_idx_str

    print("\n" + "="*30)
    print(f"Input Text: {args_cmd.text}")
    print(f"Predict   : {final_label}")
    print(f"Confidence: {result['score']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()