import argparse
import time
import evaluate
import os
import torch
from tqdm import tqdm
from optimum.intel import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline, logging as tf_logging
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from utils import load_config_as_args, setup_labels, get_logger

# 开启详细日志以查看 OpenVINO 编译过程
tf_logging.set_verbosity_info()

def main():
    parser = argparse.ArgumentParser(description="OpenVINO CPU Evaluation")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model_path", type=str, required=True, help="OV 模型文件夹")
    parser.add_argument("-n", "--num_samples", type=int, default=100, help="测试样本量")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    args_cmd = parser.parse_args()

    # 1. 加载配置 (修正顺序)
    args = load_config_as_args(args_cmd.config)
    # 必须先初始化 logger
    logger = get_logger(args.training.output_dir) 
    # 然后再传给 setup_labels
    _, label2id, id2label = setup_labels(args, logger)

    # 2. 加载数据并采样
    logger.info(f"[*] 正在从 {args.dataset.test_path} 加载数据...")
    dataset = load_dataset("json", data_files={"test": args.dataset.test_path})["test"]
    if args_cmd.num_samples < len(dataset):
        dataset = dataset.select(range(args_cmd.num_samples))
    
    # 3. 初始化 OV 模型
    logger.info(f"[*] 正在加载 OpenVINO 模型: {args_cmd.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_path)
    model = OVModelForSequenceClassification.from_pretrained(args_cmd.model_path)
    
    # 4. 创建 Pipeline (OpenVINO 在 Optimum 下自动处理 CPU 推理)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # 5. 执行推理
    input_data = KeyDataset(dataset, args.dataset.text_column)
    logger.info(f"[*] 开始评估 {len(dataset)} 条数据 (CPU - OpenVINO INT8)...")
    
    start_time = time.perf_counter()
    # list() 会触发迭代器，展示 HF 原生进度条
    results = list(classifier(input_data, batch_size=args_cmd.batch_size))
    total_time = time.perf_counter() - start_time
    
    # 6. 处理结果与映射
    preds = []
    for out in results:
        lbl = out['label']
        # 兼容 LABEL_X 和 文本标签
        if lbl.startswith("LABEL_"):
            preds.append(int(lbl.split("_")[1]))
        else:
            preds.append(label2id.get(lbl, -1))

    use_text = getattr(args.dataset, "use_text_label_flag", False)
    source_col = args.dataset.label_text_column if use_text else args.dataset.label_column
    # 确保参考标签也转换为 ID
    refs = [label2id[str(x)] for x in dataset[source_col]]

    # 7. 计算指标
    acc_metric = evaluate.load("accuracy")
    acc = acc_metric.compute(predictions=preds, references=refs)["accuracy"]

    print("\n" + " OpenVINO Result Summary ".center(50, "="))
    print(f"Throughput (TPS): {len(dataset)/total_time:.2f} samples/s")
    print(f"Total Time      : {total_time:.4f} s")
    print(f"Accuracy        : {acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()