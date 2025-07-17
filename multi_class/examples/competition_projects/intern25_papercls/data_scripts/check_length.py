import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset


def analyze_text_lengths(
    tokenizer: AutoTokenizer,
    dataset,
    text_field: str = "text",
    batch_size: int = 1000,
    threshold_lengths: list = [512, 1024]
) -> dict:
    """使用tokenizer分析文本长度统计信息
    
    Args:
        tokenizer: 分词器实例
        dataset: 加载的数据集对象
        text_field: 文本字段名
        batch_size: 批量处理大小
        threshold_lengths: 需要统计的阈值长度列表
        
    Returns:
        包含统计信息和长度数组的元组 (stats, lengths)
    """
    def tokenize_batch(examples):
        return {"tokens": tokenizer(examples[text_field])["input_ids"]}
    
    # 批量tokenize
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    # 计算长度统计
    lengths = np.array([len(tokens) for tokens in tokenized["tokens"]])
    stats = {
        "total_samples": len(lengths),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
        "avg_length": float(lengths.mean()),
        "median_length": int(np.median(lengths)),
        "percentiles": {
            p: int(np.percentile(lengths, p))
            for p in [10, 25, 50, 75, 90, 95, 99]
        },
        "threshold_counts": {
            t: int((lengths >= t).sum())
            for t in threshold_lengths
        },
        "length_distribution": defaultdict(int)
    }
    
    for length in lengths:
        stats["length_distribution"][int(length)] += 1
    
    return stats, lengths


def main():
    parser = argparse.ArgumentParser(description="文本长度统计分析工具")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="本地模型路径")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="JSONL数据文件路径")
    parser.add_argument("--text_field", type=str, default="text", 
                       help="文本字段名")
    parser.add_argument("--batch_size", type=int, default=1000, 
                       help="批量处理大小")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[512, 1024],
                       help="需要统计的阈值长度列表")
    
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 加载数据集
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    
    # 分析文本长度
    stats, lengths = analyze_text_lengths(
        tokenizer,
        dataset,
        text_field=args.text_field,
        batch_size=args.batch_size,
        threshold_lengths=args.thresholds
    )
    
    # 打印结果
    print("\n=== 基本统计 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"最大长度: {stats['max_length']}")
    print(f"最小长度: {stats['min_length']}")
    print(f"平均长度: {stats['avg_length']:.2f}")
    print(f"中位数长度: {stats['median_length']}")
    
    print("\n=== 百分位数统计 ===")
    for p, length in stats['percentiles'].items():
        print(f"{p}% 的样本长度 ≤ {length}")
    
    print("\n=== 阈值统计 ===")
    for t, count in stats['threshold_counts'].items():
        print(f"长度 ≥ {t} 的样本数: {count} ({count/stats['total_samples']:.2%})")
    
    # 打印长度分布直方图
    print("\n=== 长度分布直方图 ===")
    max_bin = min(50, stats['max_length'])
    hist, bin_edges = np.histogram(
        [min(l, max_bin) for l in lengths], 
        bins=min(20, max_bin),
        range=(0, max_bin)
    )
    for i in range(len(hist)):
        print(f"{bin_edges[i]:3.0f}-{bin_edges[i+1]:3.0f}: {'*' * int(hist[i]/max(1, hist.max())*50)} ({hist[i]})")


if __name__ == "__main__":
    main()
