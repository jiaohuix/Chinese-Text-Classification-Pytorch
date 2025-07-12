'''
@date: 2025-07-12 13:07
@author: jiaohuix
@description:
    该脚本用于将THUCNews数据集的txt格式标签文件转换为JSONL格式。
    它读取训练集、验证集和测试集的文本和标签，并将它们保存为带有特定结构的JSONL文件。
    同时，它还会处理类别映射，并为每个数据集提供数据分布的统计信息。

    使用方法:
    python scripts/thunews_preprocess.py

    确保在执行脚本前，已经准备好THUCNews数据集，并且根目录指向正确。
'''
import os
import json
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_jsonl(root_dir, output_dir):
    """
    将文本标签文件转换为JSONL格式。

    Args:
        root_dir (str): 包含train, dev, test标签文件和class文件的根目录。
        output_dir (str): 输出JSONL文件的目录。
    """

    logging.info("🚀 THUCNews 数据集预处理开始！")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载类别映射
    class_map = {}
    class_file_path = os.path.join(root_dir, 'class.txt')
    if not os.path.exists(class_file_path):
        logging.error(f"错误：未找到类别文件 '{class_file_path}'。请确保它存在。")
        return

    with open(class_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                class_map[str(i)] = class_name # 使用字符串形式的ID作为键，与输入文件匹配

    # 保存label map到JSON文件
    label_map_output_path = os.path.join(output_dir, 'label_map.json')
    with open(label_map_output_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=4)
    logging.info(f"类别映射已保存到: {label_map_output_path}")

    logging.info(f"加载的类别映射: {class_map}")

    # 处理每个数据集 (train, dev, test)
    for dataset_type in ['train', 'dev', 'test']:
        input_file_path = os.path.join(root_dir, f'{dataset_type}.txt')
        output_file_path = os.path.join(output_dir, f'{dataset_type}.jsonl')

        if not os.path.exists(input_file_path):
            logging.warning(f"警告：未找到数据集文件 '{input_file_path}'。跳过 {dataset_type}。")
            continue

        logging.info(f"正在处理数据集: {dataset_type}...")
        data_distribution = {}
        processed_count = 0
        skipped_count = 0

        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text = parts[0].strip()
                    category_id_str = parts[1].strip()

                    # 根据类别ID获取类别名称
                    category_name = class_map.get(category_id_str, "未知类别")

                    # 更新数据分布统计
                    data_distribution[category_name] = data_distribution.get(category_name, 0) + 1

                    # 创建JSON对象
                    json_data = {
                        "text": text,
                        "text_label": category_name,
                        "label": int(category_id_str),
                    }

                    # 写入JSONL文件
                    outfile.write(json.dumps(json_data, ensure_ascii=False) + '\n')
                    processed_count += 1
                else:
                    logging.warning(f"警告：跳过格式不正确的一行: {line.strip()}")
                    skipped_count += 1

        logging.info(f"已完成 {dataset_type} 数据集的转换，输出到: {output_file_path}")
        logging.info(f"{dataset_type} 数据集信息:")
        logging.info(f"  - 总共处理行数: {processed_count}")
        logging.info(f"  - 跳过错误行数: {skipped_count}")
        logging.info(f"  - 数据分布:")
        for category, count in data_distribution.items():
            logging.info(f"    - {category}: {count}")

    logging.info("✅ THUCNews 数据集预处理完成！")

if __name__ == "__main__":
    # --- 用户输入 ---
    # 请根据你的实际THUCNews数据集存储位置修改此路径
    root_directory = "../THUCNews/data/"
    # 定义输出目录，如果不存在会自动创建
    output_directory = "./data/thunews"

    # --- 用户输入结束 ---

    # 检查根目录是否存在
    if not os.path.isdir(root_directory):
        logging.error(f"错误：输入的根目录 '{root_directory}' 不存在。请检查路径是否正确。")
    else:
        convert_to_jsonl(root_directory, output_directory)
