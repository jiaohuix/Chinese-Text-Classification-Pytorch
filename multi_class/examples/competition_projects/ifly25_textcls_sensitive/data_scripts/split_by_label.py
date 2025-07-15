import os
from datasets import load_dataset

def split_jsonl_by_label(input_file, output_dir, seed=42):
    # 加载数据集
    dataset = load_dataset('json', data_files=input_file)['train']
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有不同的text_label值
    unique_labels = set(dataset['text_label'])
    
    # 为每个text_label创建一个数据集分片
    for label in unique_labels:
        # 过滤出该标签的数据
        filtered_data = dataset.filter(lambda example: example['text_label'] == label)
        
        # 对过滤后的数据进行shuffle（随机打乱）
        shuffled_data = filtered_data.shuffle(seed=seed)

        # 构建输出文件名，使用label名称并替换可能的非法文件名字符
        safe_label = label.replace('/', '_').replace('\\', '_').replace(':', '_') \
                          .replace('*', '_').replace('?', '_').replace('"', '_') \
                          .replace('<', '_').replace('>', '_').replace('|', '_')
        output_file = os.path.join(output_dir, f"{safe_label}.jsonl")
        
        # 将过滤后的数据保存到对应的jsonl文件
        filtered_data.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
        print(f"已输出 {len(filtered_data)} 条记录到 {output_file}")

if __name__ == "__main__":
    # 设置输入文件路径和输出目录
    input_jsonl = "data/processed/train.jsonl"  # 替换为实际输入文件路径
    output_directory = "data/processed/output_by_label"  # 替换为实际输出目录
    
    # 执行分割
    split_jsonl_by_label(input_jsonl, output_directory)    