import argparse
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer
from datasets import load_dataset


def analyze_text_lengths(
    tokenizer: AutoTokenizer,
    dataset,
    text_field: str = "text",
    batch_size: int = 1000
) -> dict:
    """使用tokenizer分析文本长度统计信息
    
    Args:
        tokenizer: 分词器实例
        dataset: 加载的数据集对象
        text_field: 文本字段名
        batch_size: 批量处理大小
        
    Returns:
        包含统计信息的字典
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
    lengths = [len(tokens) for tokens in tokenized["tokens"]]
    stats = {
        "total_samples": len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "length_distribution": defaultdict(int)
    }
    
    for length in lengths:
        stats["length_distribution"][length] += 1
    
    return stats


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
    
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 加载数据集
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    
    # 分析文本长度
    stats = analyze_text_lengths(
        tokenizer,
        dataset,
        text_field=args.text_field,
        batch_size=args.batch_size
    )
    
    # 打印结果
    print(f"总样本数: {stats['total_samples']}")
    print(f"最大长度: {stats['max_length']}")
    print(f"最小长度: {stats['min_length']}")
    print(f"平均长度: {stats['avg_length']:.2f}")
    # print("\n长度分布统计:")
    # for length, count in sorted(stats["length_distribution"].items()):
    #     print(f"{length} tokens: {count} samples")


if __name__ == "__main__":
    main()
