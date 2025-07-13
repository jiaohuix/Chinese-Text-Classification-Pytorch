#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将对话JSONL数据转换为BERT分类训练格式
支持命令行参数配置输入输出路径
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对话数据转BERT分类格式')
    parser.add_argument('-i', '--input', required=True, 
                        help='输入JSONL文件路径')
    parser.add_argument('-o', '--output', required=True,
                        help='输出JSONL文件路径')
    parser.add_argument('--force_ascii', action='store_true',
                        help='强制ASCII编码输出')
    return parser.parse_args()


def ensure_directory_exists(file_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(file_path)
    if output_dir:  # 非空路径才需要创建
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def load_and_validate_data(input_file):
    """加载并验证输入数据"""
    try:
        dataset = load_dataset("json", data_files=input_file, split="train")
        print(f"✅ 成功加载数据，共 {len(dataset)} 条样本")
        print(f"📊 数据结构: {dataset}")
        print(f"🏷️ 列名: {dataset.column_names}")
        
        # 验证必要字段存在
        if "conversation" not in dataset.column_names:
            raise ValueError("数据中缺少必需的 'conversation' 字段")
            
        return dataset
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        raise


def convert_to_bert_format(example):
    """
    将单条对话数据转换为BERT分类格式
    Args:
        example: 包含conversation字段的原始数据
    Returns:
        dict: 包含text, text_label, label的字典
    """
    # 生成A-Z的标签映射 (26个类别)
    text_labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    lbl_map = {lbl: idx for idx, lbl in enumerate(text_labels)}
    
    # 提取对话内容 (假设conversation是单条对话的列表)
    conversation = example["conversation"][0]  # 取第一条对话
    input_text = conversation["human"]        # 用户输入
    output_label = conversation["assistant"]  # 助手回复标签
    
    # 验证标签有效性
    if output_label not in lbl_map:
        raise ValueError(f"非法标签: {output_label}。允许的标签范围: A-Z")
    
    return {
        "text": input_text,            # 输入文本
        "text_label": output_label,    # 原始字母标签
        "label": int(lbl_map[output_label])  # 数字标签(0-25)
    }


def process_dataset(dataset):
    """执行数据集转换"""
    try:
        print("🔄 开始转换数据格式...")
        bert_ds = dataset.map(
            convert_to_bert_format,
            remove_columns=dataset.column_names,  # 移除原始列
            batched=False,                        # 逐条处理
            desc="Processing"
        )
        print("✅ 数据转换完成")
        return bert_ds
    except Exception as e:
        print(f"❌ 数据转换失败: {str(e)}")
        raise


def save_results(dataset, output_file, force_ascii):
    """保存转换后的数据集"""
    try:
        ensure_directory_exists(output_file)
        print(f"💾 正在保存到 {output_file}...")
        
        dataset.to_json(
            output_file,
            orient="records",
            lines=True,
            force_ascii=force_ascii
        )
        
        print(f"✅ 成功保存 {len(dataset)} 条样本")
        print(f"📄 输出文件: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")
        raise


def main():
    """主处理流程"""
    args = parse_arguments()
    
    try:
        # 1. 加载数据
        dataset = load_and_validate_data(args.input)
        
        # 2. 转换格式
        bert_dataset = process_dataset(dataset)
        
        # 3. 保存结果
        save_results(bert_dataset, args.output, args.force_ascii)
        
        # 4. 打印样例
        print("\n🔍 转换后样例:")
        print(json.dumps(bert_dataset[0], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"⛔ 处理中断: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
