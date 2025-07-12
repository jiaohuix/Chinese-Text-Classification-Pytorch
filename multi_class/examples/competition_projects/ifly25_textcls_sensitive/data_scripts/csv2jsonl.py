import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_label_map(label_map_file: str) -> Dict[str, int]:
    """加载已有的标签映射"""
    with open(label_map_file, 'r', encoding='utf-8') as f:
        return json.load(f)  # 直接加载为字典格式

def get_label_statistics(df: pd.DataFrame, text_label_col: str) -> Dict[str, Tuple[int, float]]:
    """获取每个标签的统计信息（数量和占比）"""
    label_counts = df[text_label_col].value_counts()
    total = len(df)
    return {label: (count, count/total) for label, count in label_counts.items()}

def convert_csv_to_jsonl(
    input_file: str,
    output_file: str,
    text_col: str = 'text',
    text_label_col: Optional[str] = 'text_label',
    separator: str = ',',
    save_labels: Optional[str] = None,
    test_mode: bool = False,
    label_map_file: Optional[str] = None,
    random_seed: Optional[int] = None
) -> None:
    """
    将CSV文件转换为JSONL格式，支持训练和测试两种模式。

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出JSONL文件路径
        text_col: 文本列名
        text_label_col: 文本标签列名（可选）
        separator: CSV分隔符
        save_labels: 保存标签映射的文件名（训练模式）
        test_mode: 是否为测试模式
        label_map_file: 测试模式下使用的标签映射文件
        random_seed: 随机种子，用于训练集shuffle
    """
    # 读取CSV文件
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file, sep=separator)
    
    # 验证文本列是否存在
    if text_col not in df.columns:
        raise ValueError(f"Missing required text column: {text_col}")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理标签映射
    label_map = {}
    if test_mode:
        if not label_map_file:
            raise ValueError("Label map file is required in test mode")
        print(f"Loading label mapping from: {label_map_file}")
        label_map = load_label_map(label_map_file)
    elif text_label_col and text_label_col in df.columns:
        # 训练模式，构建新的标签映射
        unique_labels = sorted(df[text_label_col].unique())
        label_map = {text: idx for idx, text in enumerate(unique_labels)}
        
        # 获取标签统计信息
        label_stats = get_label_statistics(df, text_label_col)
        
        # 保存标签映射到输出目录
        if save_labels:
            save_labels_path = output_dir / save_labels
            print(f"Saving label mapping to: {save_labels_path}")
            with open(save_labels_path, 'w', encoding='utf-8') as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
        
        # 打印标签映射和统计信息
        print(f"\nFound {len(label_map)} unique labels:")
        for text, idx in label_map.items():
            count, percentage = label_stats[text]
            print(f"  {text} -> {idx} ({count} samples, {percentage:.2%})")
    
    # 准备数据
    records = []
    for _, row in df.iterrows():
        data = {"text": row[text_col]}
        
        # 如果有标签列且存在该标签
        if text_label_col and text_label_col in df.columns:
            text_label = row[text_label_col]
            if text_label in label_map:
                data.update({
                    "label": label_map[text_label],
                    "text_label": text_label
                })
            else:
                print(f"Warning: Unknown label '{text_label}' found in data")
        
        records.append(data)
    
    # 非测试模式且指定了随机种子时，打乱数据顺序
    if not test_mode and random_seed is not None:
        print(f"\nShuffling data with random seed: {random_seed}")
        np.random.seed(random_seed)
        np.random.shuffle(records)
    
    # 保存为JSONL格式
    print(f"\nSaving to JSONL: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed. Processed {len(records)} records.")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to JSONL format with label mapping')
    parser.add_argument('-i', '--input', default='train.csv', help='Input CSV file path (default: train.csv)')
    parser.add_argument('-o', '--output', default='train.jsonl', help='Output JSONL file path (default: train.jsonl)')
    parser.add_argument('-t', '--text_col', default='text', help='Text column name (default: text)')
    parser.add_argument('-l', '--text_label_col', default='text_label', help='Text label column name (default: text_label)')
    parser.add_argument('-s', '--sep', default=',', help='CSV separator (default: ,)')
    parser.add_argument('-d', '--save_labels', default='labels.json', help='Label mapping file name (default: labels.json)')
    parser.add_argument('--test', action='store_true', help='Enable test mode')
    parser.add_argument('--label_map', help='Path to existing label mapping JSON file (required in test mode)')
    parser.add_argument('--seed', type=int, help='Random seed for shuffling training data (default: None, no shuffle)')
    
    args = parser.parse_args()
    
    try:
        convert_csv_to_jsonl(
            args.input,
            args.output,
            args.text_col,
            args.text_label_col,
            args.sep,
            args.save_labels,
            args.test,
            args.label_map,
            args.seed
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
