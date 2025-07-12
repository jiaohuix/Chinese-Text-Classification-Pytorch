import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import pandas as pd

def load_model(model_dir):
    """
    加载模型和tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    classifier = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=0,
        truncation=True, 
        max_length=256
    )
    return classifier

def load_label_map(label_file):
    """
    加载标签映射文件
    """
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    label_map = {int(item['id']): item['text'] for item in label_data['labels']}
    return label_map

def predict(input_file, output_file, model_dir, label_file=None):
    """
    执行预测并保存结果
    """
    # 加载模型
    classifier = load_model(model_dir)
    
    # 加载标签映射
    if label_file:
        label_map = load_label_map(label_file)
    else:
        # 如果没有提供标签文件，假设模型输出就是文本标签
        label_map = None
    
    # 加载输入数据
    dataset = load_dataset('json', data_files=input_file, split='train')
    texts = dataset['text']
    
    # 执行预测
    results = classifier(texts)
    
    # 处理预测结果
    predictions = []
    for idx, result in enumerate(results):
        if label_map:
            # 将数字标签转换为文本标签
            label_id = int(result['label'])
            label_text = label_map[label_id]
        else:
            # 直接使用模型输出的文本标签
            label_text = result['label']
        
        predictions.append({
            'id': idx,
            '类别': label_text
        })
    
    # 保存为CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python predict.py <输入jsonl文件> <输出csv文件> <模型目录> [标签文件]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_dir = sys.argv[3]
    label_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    predict(input_file, output_file, model_dir, label_file)
