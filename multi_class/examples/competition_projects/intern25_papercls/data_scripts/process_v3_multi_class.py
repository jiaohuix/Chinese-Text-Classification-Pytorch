from collections import Counter
import random
import json
from typing import List
import numpy as np
from datasets import load_dataset, Dataset

# --- Helper Function to Get Category Mapping ---
def extract_category_mapping():
    """定义类别到选项的映射"""
    category_to_option = {
        'quant-ph': 'A',
        'physics.chem-ph': 'B',
        'physics.atom-ph': 'C',
        'cond-mat.soft': 'D',
        'cs.RO': 'E',
        'cs.CL': 'F',
        'cs.SE': 'G',
        'cs.IR': 'H',
        'hep-th': 'I',
        'hep-ph': 'J',
        'physics.optics': 'K',
        'cs.AI': 'L',
        'cs.CV': 'M',
        'nucl-th': 'N',
        'astro-ph': 'O',
        'math.PR': 'P',
        'cs.OS': 'Q',
        'eess.SP': 'R',
        'math.OC': 'S',
        'math.DS': 'T',
        'math.DG': 'U',
        'math.MP': 'V',
        'cs.MM': 'W',
        'stat.ME': 'X',
        'math.CO': 'Y',
        'cs.NE': 'Z'
    }
    return category_to_option

# --- Configuration Parameters ---
DATA_FILE = "../data_arxiv/arxiv-metadata-oai-snapshot.json"
# DATA_FILE = "../data_arxiv/corpus.jsonl"


# 获取类别到选项的映射
LABEL_MAP = extract_category_mapping()
NEW_LABELS = list(LABEL_MAP.keys())


RANDOM_SEED = 42
NUM_PROCESSES = 16 # 根据你的CPU核心数调整

FORCE_ASCII = False  # Set to True if ASCII encoding is required


# SAMPLES_PER_LABEL = 500
SAMPLES_PER_LABEL = 600
# HANDLE_MULTI_LABELS = False
# FILTERED_OUTPUT_FILE = "data/filtered_data.jsonl"
# SAMPLED_OUTPUT_FILE = "data/sample_data_600.jsonl"
# SAMPLES_PER_LABEL = 1100
# FILTERED_OUTPUT_FILE = "data/filtered_data3_wo_sample.jsonl"
# SAMPLED_OUTPUT_FILE = "data/sample_data_1100.jsonl"


HANDLE_MULTI_LABELS = True
SAMPLES_PER_LABEL = 600
FILTERED_OUTPUT_FILE = "data/mclass/filtered_data.jsonl"
SAMPLED_OUTPUT_FILE = "data/mclass/sample_data_600.jsonl"


# !! 重要 !! 更新你的PROMPT_TEMPLATE，确保包含所有26个类别的选项
# 注意：请确保PROMPT_TEMPLATE中的选项代码（A, B, C...）与LABEL_MAP中的值一致
# 并且所有26个类别都已列出。
PROMPT_TEMPLATE = """Based on the title '{title}', authors '{authors}', and abstract '{abstract}', please determine the scientific category of this paper. Additional info: {additional_info}.
"""

# 动态生成分类选项，这里直接使用 LABEL_MAP
for label, code in LABEL_MAP.items():
    PROMPT_TEMPLATE += f"{code}. {label}\n"

PROMPT_TEMPLATE += "\n" # 添加一个空行

SYSTEM_PROMPT = "你是个优秀的论文分类师"


# --- Functions (保持不变，除了上面提到的处理逻辑) ---

def save_dataset_to_jsonl(dataset, output_file, force_ascii):
    """Saves the dataset to a JSONL file."""
    if dataset is None:
        print("Dataset is None, cannot save.")
        return

    print(f"Saving dataset to {output_file}...")
    try:
        dataset.to_json(output_file, orient="records", lines=True, force_ascii=force_ascii)
        print(f"Data successfully saved to {output_file} in JSONL format.")
    except Exception as e:
        print(f"Error saving dataset to {output_file}: {e}")


def load_and_filter_dataset(
    data_file: str,
    labels: List[str],
    label_column: str,
    handle_multi_labels: bool = False # 新增参数，默认 False
):
    """Loads the dataset and filters it based on the given labels."""
    print(f"Loading dataset from: {data_file}")
    try:
        ds = load_dataset("json", data_files=data_file, split="train", num_proc=NUM_PROCESSES)
        print(f"Initial dataset size: {len(ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


    def filter_func(example):
        label = example.get(label_column)
        if label in labels:
            return True
        # print(f"[!] 警告: 在新字典中找不到 {label}, 跳过该条数据")
        return False

    # 先map过滤掉无关的标签
    def map_func(example):
        # 保留有效的标签，然后拼接多个标签（如果拼接后只有一个保留）
        label = example.get(label_column)
        if label in labels:
            return example
        else:
            split_labels = [lbl.strip() for lbl in label.split(" ")]
            valid_labels = [lbl for lbl in split_labels if lbl in labels]
            new_lbl = " ".join(valid_labels)
            example[label_column] = new_lbl
            # print(f"[!]valid labels:  {new_lbl}")
            return example

    # shuffle
    ds = ds.shuffle(seed=42)
    # ds = ds.add_column("single_class", [False] * len(ds))

    if not handle_multi_labels:
        print(f"Filtering dataset to include labels: {labels[:5]}...") # 打印部分标签
        ds_filtered = ds.filter(filter_func, num_proc=NUM_PROCESSES)
        print(f"Filtered dataset size: {len(ds_filtered)}")
    else:
        print(f"Filtering dataset to include labels: {labels[:5]}...") # 打印部分标签
        ds_filtered = ds.map(map_func, num_proc=NUM_PROCESSES)
        ds_filtered = ds_filtered.filter(filter_func, num_proc=NUM_PROCESSES)
        print(f"Filtered dataset size: {len(ds_filtered)}")

    return ds_filtered


def process_data(example, prompt_template, label_map):
    """Processes each example to create the input and output prompts."""
    title = example.get("title", "N/A")
    authors = example.get("authors", "N/A")
    abstract = example.get("abstract", "N/A")
    comments = example.get("comments", "")
    doi = example.get("doi", "N/A")
    category = example.get("categories")

    if isinstance(authors, list):
        authors = ", ".join(authors)

    additional_info = f"Comments: {comments} DOI: {doi}"

    try:
        prompt = prompt_template.format(title=title, authors=authors, abstract=abstract,
                                         additional_info=additional_info)
    except KeyError as e:
        print(f"Error formatting prompt: Missing key {e} in example: {example}")
        return None

    if category not in label_map:
        print(f"Error: Category '{category}' not found in LABEL_MAP.")
        return None

    example["human"] = prompt
    example["assistant"] = label_map[category]

    return example


def create_conversation_dataset(dataset, system_prompt):
    """Creates the final dataset with the conversation format."""
    if dataset is None:
        print("Dataset is None, cannot create conversation dataset.")
        return None

    print("Creating conversation dataset...")
    data_ls = []
    for item in dataset:
        ipt = item.get("human")
        out = item.get("assistant")
        if ipt and out:
            conversation = [{"human": ipt, "assistant": out}]

            data_ls.append({"system": system_prompt, "conversation": conversation})
        else:
            print(f"Skipping item due to missing input or output: {item}")
    return Dataset.from_list(data_ls)


def sample_dataset(dataset, label_col, k, seed=42):
    np.random.seed(seed)  # 设置随机种子保证可复现性
    labels = dataset[label_col]  # 获取所有样本的标签列表
    unique_labels = list(set(labels))  # 获取唯一标签类别
    
    indices = []  # 用于存储最终选中的样本索引
    
    # 遍历每个类别
    for label in unique_labels:
        # 找出当前类别的所有样本索引
        label_indices = [i for i, x in enumerate(labels) if x == label]
        # 从当前类别中随机采样k个（不超过类别样本数）
        sampled = np.random.choice(label_indices, size=min(k, len(label_indices)), replace=False)
        indices.extend(sampled.tolist())  # 添加到总索引列表
    
    return dataset.select(indices)  # 根据索引选择样本


def main():
    """Main function to execute the data processing pipeline."""
    # 1. Load and filter the dataset
    filtered_ds = load_and_filter_dataset(DATA_FILE, NEW_LABELS, "categories", handle_multi_labels=HANDLE_MULTI_LABELS)

    if filtered_ds is None:
        print("Failed to load or filter dataset. Exiting.")
        return

    print("filtered_ds len:", len(filtered_ds))
    # 保存过滤的全量数据
    save_dataset_to_jsonl(filtered_ds, FILTERED_OUTPUT_FILE, FORCE_ASCII)

    # 2. Sample the dataset
    sampled_ds = sample_dataset(filtered_ds, label_col="categories", k=SAMPLES_PER_LABEL, seed=RANDOM_SEED)

    if sampled_ds is None:
        print("Failed to sample dataset. Exiting.")
        return

    # filter去重
    SEEN_TITLES = sampled_ds.unique("title")
    print("SEEN_TITLES", len(SEEN_TITLES))

    # Define a function to check if a title is in the seen_titles set
    def is_title_seen(example):
        title = example.get("title")
        return title not in SEEN_TITLES
    # Apply the filter
    print("before filt",filtered_ds)
    filtered_ds = filtered_ds.filter(is_title_seen, num_proc=NUM_PROCESSES)
    print("after filt",filtered_ds)


    # 3. Process the data to create input and output prompts
    print("Processing sampled data to create prompts...")
    sampled_ds = sampled_ds.map(
        lambda example: process_data(example, PROMPT_TEMPLATE, LABEL_MAP),
        num_proc=NUM_PROCESSES,
        batched=False
    )
    print("过滤空数据...")
    sampled_ds = sampled_ds.filter(lambda x: x is not None)

    # 保存采样的数据
    print("保存采样数据...")
    sampled_ds_chat = create_conversation_dataset(sampled_ds, SYSTEM_PROMPT)
    save_dataset_to_jsonl(sampled_ds_chat, SAMPLED_OUTPUT_FILE, FORCE_ASCII)

    # 保存过滤的数据
    FILTER_CHAT_FILE = FILTERED_OUTPUT_FILE.replace(".jsonl","_chat.jsonl")
    filtered_ds_chat = filtered_ds.map(
        lambda example: process_data(example, PROMPT_TEMPLATE, LABEL_MAP),
        num_proc=NUM_PROCESSES,
        batched=False
    )
    filtered_ds_chat =  create_conversation_dataset(filtered_ds_chat, SYSTEM_PROMPT)
    save_dataset_to_jsonl(filtered_ds_chat, FILTER_CHAT_FILE, FORCE_ASCII)

    print("\n--- Data processing pipeline completed. ---")


if __name__ == "__main__":
    main()
