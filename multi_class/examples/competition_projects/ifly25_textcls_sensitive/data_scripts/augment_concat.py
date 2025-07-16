import json
import random
from collections import defaultdict
import math
import os
from typing import List, Dict, Any, Set

def count_label_distribution(data: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    统计数据集中各个类别的分布。

    Args:
        data: 包含字典的列表，每个字典代表一条数据，包含 'label' 键。

    Returns:
        键为类别标签，值为该类别在数据集中出现的次数。
    """
    distribution = defaultdict(int)
    for item in data:
        distribution[item['label']] += 1
    return distribution

def augment_data_by_category(input_file: str, max_per_category: int, num_splices: int, output_file: str):
    """
    根据类别对文本分类的 JSONL 数据集进行增强。

    Args:
        input_file: 输入的 JSONL 文件路径。
        max_per_category: 每个类别允许的最大样本数量（包括原始样本和增强样本）。
        num_splices: 每次增强时，从同一类别中随机选择的文本数量。
                           例如，设置为 2 表示将随机选择两个文本进行拼接。
        output_file: 输出的增强后的 JSONL 文件路径。
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在。")
        return

    if num_splices < 2:
        print("警告：num_splices 应至少为 2 以进行拼接操作。")
        num_splices = 2 # 强制设置为至少2，以进行拼接

    print(f"开始处理文件: {input_file}")
    print(f"每个类别的最大数量: {max_per_category}")
    print(f"每次拼接选择的文本数量: {num_splices}")

    # --- 读取原始数据 ---
    original_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                original_data.append(json.loads(line))
    except Exception as e:
        print(f"读取文件 '{input_file}' 时发生错误: {e}")
        return

    # --- 按类别分组数据 ---
    data_by_category = defaultdict(list)
    for item in original_data:
        data_by_category[item['label']].append(item)

    # --- 统计增强前的数据分布 ---
    original_distribution = count_label_distribution(original_data)
    print("\n--- 增强前类别分布 ---")
    for label, count in sorted(original_distribution.items()):
        print(f"类别 {label}: {count} 个样本")

    augmented_data = []
    # 用于去重检查的集合，存储已生成的文本
    generated_texts: Set[str] = set()

    for label, samples in data_by_category.items():
        current_label_data = list(samples) # 复制一份，避免修改原始列表
        current_count = len(current_label_data)

        print(f"\n处理类别 {label} (当前数量: {current_count})")

        # 如果当前类别数量已经超过最大限制，则不进行增强
        if current_count >= max_per_category:
            print(f"类别 {label} 已达到或超过最大数量 ({max_per_category})，跳过增强。")
            augmented_data.extend(current_label_data)
            # 将原始样本的文本添加到 generated_texts 中，防止生成重复
            for sample in current_label_data:
                generated_texts.add(sample['text'])
            continue

        # --- 计算还需要生成多少样本 ---
        samples_needed = max_per_category - current_count
        if samples_needed <= 0: # 再次确认，以防万一
            print(f"类别 {label} 已达到最大数量，无需生成更多样本。")
            augmented_data.extend(current_label_data)
            for sample in current_label_data:
                generated_texts.add(sample['text'])
            continue

        # --- 准备用于拼接的样本池 ---
        # 创建一个副本，避免在采样时修改原始列表
        available_samples_for_splice = list(current_label_data)

        # 记录这次类别生成的样本数
        newly_generated_count = 0
        # 增加一个最大尝试次数，防止因特殊情况（如极少样本，或多次生成重复文本）导致无限循环
        max_attempts_per_category = samples_needed * num_splices * 5 # 粗略估计，可调

        for _ in range(max_attempts_per_category):
            # 检查是否已经生成了足够的样本
            if newly_generated_count >= samples_needed:
                break

            # 确保有足够的样本可供选择进行拼接
            if len(available_samples_for_splice) < num_splices:
                print(f"类别 {label} 的可用样本 ({len(available_samples_for_splice)}) 不足以进行 {num_splices} 个的拼接，停止该类别的增强。")
                break

            # --- 随机选择 num_splices 个样本进行拼接 ---
            # 使用 random.sample 以确保选择的样本是唯一的
            try:
                chosen_samples = random.sample(available_samples_for_splice, num_splices)
            except ValueError: # 如果available_samples_for_splice < num_splices
                print(f"类别 {label} 样本池不足，无法进行 {num_splices} 个的采样。")
                break


            # --- 拼接文本 ---
            # 注意：实际应用中可能需要更智能的拼接逻辑，例如考虑句子边界等
            concatenated_text = " ".join([sample['text'] for sample in chosen_samples])

            # --- 去重检查 ---
            if concatenated_text in generated_texts:
                # print(f"已生成重复文本，跳过: '{concatenated_text[:30]}...'") # 调试用
                continue

            # --- 创建新的增强样本 ---
            new_sample = {
                "text": concatenated_text,
                "label": label,
                # 假设所有样本的 text_label 都相同，取第一个的。如果不同，需要根据业务逻辑调整。
                "text_label": chosen_samples[0]['text_label']
            }

            current_label_data.append(new_sample)
            generated_texts.add(concatenated_text)
            newly_generated_count += 1

        # --- 处理当前类别最终的样本数量 ---
        # 确保总样本数量不超过 max_per_category
        final_label_data = current_label_data[:max_per_category]
        augmented_data.extend(final_label_data)

        print(f"类别 {label} 增强完成，共生成 {newly_generated_count} 个新样本，最终拥有 {len(final_label_data)} 个样本。")

    # --- 统计增强后的数据分布 ---
    final_distribution = count_label_distribution(augmented_data)
    print("\n--- 增强后类别分布 ---")
    for label, count in sorted(final_distribution.items()):
        print(f"类别 {label}: {count} 个样本")

    # --- 写入增强后的数据到输出文件 ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in augmented_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\n数据增强完成！增强后的数据已保存至: {output_file}")
    except Exception as e:
        print(f"写入文件 '{output_file}' 时发生错误: {e}")



if __name__ == "__main__":
    # 定义你的参数
    INPUT_JSONL_FILE = "data/processed/train.jsonl" # 替换为你的输入文件路径
    MAX_SAMPLES_PER_CATEGORY = 8000              # 替换为你想要的每个类别的最大数量
    SPLICE_COUNT = 2                           # 替换为你想要的每次拼接选择的文本数量 (例如 2 或 3)
    OUTPUT_JSONL_FILE = "data/processed/train_cat8k_splice2.jsonl" # 替换为你想要的输出文件路径

    # 调用增强函数
    augment_data_by_category(INPUT_JSONL_FILE, MAX_SAMPLES_PER_CATEGORY, SPLICE_COUNT, OUTPUT_JSONL_FILE)
    