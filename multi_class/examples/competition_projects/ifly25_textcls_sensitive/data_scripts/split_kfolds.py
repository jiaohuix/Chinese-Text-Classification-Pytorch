'''
将JSONL文件拆分为N份交叉验证数据集。

用法：
python split_cv.py -i input.jsonl -o output_dir [-n 5] [--seed 42]

参数说明：
-i/--input: 输入的JSONL文件
-o/--output: 输出目录，将在其下创建fold_1, fold_2...等子目录
-n/--num_folds: 交叉验证份数，默认为5
--seed: 随机种子，默认为42

示例：
python data_scripts/split_kfolds.py -i data/processed/train.jsonl -o data/kfolds -n 5 --seed 42
'''

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存为JSONL格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_cv_splits(
    data: List[Dict[str, Any]],
    output_dir: str,
    num_folds: int = 5,
    random_seed: int = 42
):
    """
    创建交叉验证数据集分割。
    
    Args:
        data: 数据列表
        output_dir: 输出目录
        num_folds: 交叉验证份数
        random_seed: 随机种子
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 计算每份的大小
    total_size = len(data)
    fold_size = total_size // num_folds
    
    # 随机打乱数据
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    
    # 创建每个fold的目录并保存数据
    for fold_idx in range(num_folds):
        # 计算当前fold的验证集范围
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size if fold_idx < num_folds - 1 else total_size
        
        # 获取验证集和训练集的索引
        dev_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([
            indices[:start_idx],
            indices[end_idx:]
        ])
        
        # 创建当前fold的目录
        fold_dir = Path(output_dir) / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练集和验证集
        train_data = [data[i] for i in train_indices]
        dev_data = [data[i] for i in dev_indices]
        
        train_file = fold_dir / "train.jsonl"
        dev_file = fold_dir / "dev.jsonl"
        
        save_jsonl(train_data, train_file)
        save_jsonl(dev_data, dev_file)
        
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train size: {len(train_data)}")
        print(f"  Dev size: {len(dev_data)}")
        
        # 打印每个标签在训练集和验证集中的分布
        if 'label' in data[0]:
            train_dist = get_label_distribution(train_data)
            dev_dist = get_label_distribution(dev_data)
            
            print("\n  Label distribution:")
            print("  Train:")
            for label, count in train_dist.items():
                print(f"    Label {label}: {count} ({count/len(train_data):.2%})")
            print("  Dev:")
            for label, count in dev_dist.items():
                print(f"    Label {label}: {count} ({count/len(dev_data):.2%})")
        print()

def get_label_distribution(data: List[Dict[str, Any]]) -> Dict[int, int]:
    """获取数据集中标签的分布"""
    dist = {}
    for item in data:
        if 'label' in item:
            label = item['label']
            dist[label] = dist.get(label, 0) + 1
    return dict(sorted(dist.items()))

def main():
    parser = argparse.ArgumentParser(description='Split JSONL file into N-fold cross validation sets')
    parser.add_argument('-i', '--input', default="train.jsonl", help='Input JSONL file')
    parser.add_argument('-o', '--output', default="cv_splits", help='Output directory')
    parser.add_argument('-n', '--num_folds', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print(f"Loading data from {args.input}")
        data = load_jsonl(args.input)
        print(f"Loaded {len(data)} records")
        
        # 创建交叉验证分割
        print(f"\nCreating {args.num_folds}-fold cross validation splits")
        create_cv_splits(
            data,
            args.output,
            args.num_folds,
            args.seed
        )
        
        print(f"\nSplit completed. Results saved in: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main() 