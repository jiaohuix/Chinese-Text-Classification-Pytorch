import argparse
import time
import torch
import evaluate
import json
import os
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from utils import load_config_as_args, setup_labels, get_logger

def main():
    parser = argparse.ArgumentParser(description="NLP 分类模型评估脚本 (Torch 版)")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="模型文件夹路径")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="推理批大小")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu", "cuda"], default=None, 
                        help="覆盖设备配置 (cpu/gpu)")
    parser.add_argument("-n", "--num_samples", type=int, default=None, help="限制评估样本数量")
    args_cmd = parser.parse_args()

    # 1. 配置加载 (修正顺序：先加载 config，再定义 logger)
    args = load_config_as_args(args_cmd.config)
    logger = get_logger(args.training.output_dir)
    _, label2id, id2label = setup_labels(args, logger)

    # 2. 设备逻辑
    device_req = args_cmd.device or getattr(args.inference, "device", "cpu")
    if device_req in ["gpu", "cuda"] and torch.cuda.is_available():
        device_idx = 0
        device_name = "GPU (cuda:0)"
    else:
        device_idx = -1
        device_name = "CPU"
    logger.info(f"[*] 当前使用推理设备: {device_name}")

    # 3. 加载数据集
    logger.info(f"[*] 正在加载测试集: {args.dataset.test_path}")
    dataset = load_dataset("json", data_files={"test": args.dataset.test_path})["test"]
    
    # 限制样本量 (与 OV 脚本对齐)
    if args_cmd.num_samples is not None and args_cmd.num_samples < len(dataset):
        dataset = dataset.select(range(args_cmd.num_samples))
        logger.info(f"[*] 已采样前 {args_cmd.num_samples} 条数据进行评测")

    # 4. 初始化 Pipeline
    tokenizer = AutoTokenizer.from_pretrained(args_cmd.model_path)
    classifier = pipeline(
        "text-classification",
        model=args_cmd.model_path,
        tokenizer=tokenizer,
        device=device_idx,
        truncation=True,
        max_length=128
    )

    # 5. 数据清洗
    raw_texts = dataset[args.dataset.text_column]
    texts = [str(t) if t is not None else "" for t in raw_texts]

    # 6. 执行推理
    logger.info(f"[*] 开始推理 (样本量: {len(texts)}, Batch Size: {args_cmd.batch_size})...")
    start_time = time.perf_counter() # 使用更精确的 perf_counter
    
    results = []
    for out in tqdm(classifier(texts, batch_size=args_cmd.batch_size), total=len(texts), desc="Torch 推理"):
        results.append(out)
    
    total_time = time.perf_counter() - start_time
    fps = len(texts) / total_time

    # 7. 结果解析
    preds_ids = []
    pred_labels_text = []
    for out in results:
        label_raw = out['label']
        if label_raw.startswith("LABEL_"):
            p_id = int(label_raw.split("_")[1])
            preds_ids.append(p_id)
            pred_labels_text.append(id2label.get(p_id, label_raw))
        else:
            p_id = label2id.get(label_raw, -1)
            preds_ids.append(p_id)
            pred_labels_text.append(label_raw)

    use_text = getattr(args.dataset, "use_text_label_flag", False)
    source_col = args.dataset.label_text_column if use_text else args.dataset.label_column
    true_ids = [label2id[str(x)] for x in dataset[source_col]]

    # 8. 指标计算
    acc_metric = evaluate.load("accuracy")
    acc_res = acc_metric.compute(predictions=preds_ids, references=true_ids)

    # 9. 输出汇总
    print("\n" + " Torch 评估报告 ".center(50, "="))
    print(f"模型路径   : {args_cmd.model_path}")
    print(f"推理设备   : {device_name}")
    print(f"总样本量   : {len(texts)}")
    print(f"处理耗时   : {total_time:.4f} s")
    print(f"吞吐量     : {fps:.2f} samples/s")
    print(f"准确率     : {acc_res['accuracy']:.4f}")
    print("="*50)

    # 导出 Bad Cases
    bad_cases_path = os.path.join(args.training.output_dir, "torch_bad_cases.jsonl")
    with open(bad_cases_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            if preds_ids[i] != true_ids[i]:
                item = {
                    "text": texts[i],
                    "gold": id2label[true_ids[i]],
                    "pred": pred_labels_text[i]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"[*] 评估完成，Bad Cases 已保存")

if __name__ == "__main__":
    main()