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
# DATA_FILE = "../data_arxiv/corpus2.jsonl"


# 获取类别到选项的映射
LABEL_MAP = extract_category_mapping()
NEW_LABELS = list(LABEL_MAP.keys())


RANDOM_SEED = 42
NUM_PROCESSES = 16 # 根据你的CPU核心数调整
random.seed(RANDOM_SEED)  # You can use any integer value as your seed
FORCE_ASCII = False  # Set to True if ASCII encoding is required


# SAMPLES_PER_LABEL = 500
SAMPLES_PER_LABEL = 600 # 500训练 100评估
HANDLE_MULTI_LABELS = False
FILTERED_OUTPUT_FILE = "data/filtered_data.jsonl"
SAMPLED_OUTPUT_FILE = "data/sample_data_600_multi_template.jsonl"

# FILTERED_OUTPUT_FILE = "data/filtered_datatst.jsonl"
# SAMPLED_OUTPUT_FILE = "data/sample_data_60600tst.jsonl"


# HANDLE_MULTI_LABELS = True
# FILTERED_OUTPUT_FILE = "data/filtered_data_m.jsonl"
# SAMPLED_OUTPUT_FILE = "data/sample_data_500_m.jsonl"


# !! 重要 !! 更新你的PROMPT_TEMPLATE，确保包含所有26个类别的选项
# 注意：请确保PROMPT_TEMPLATE中的选项代码（A, B, C...）与LABEL_MAP中的值一致
# 并且所有26个类别都已列出。
PROMPT_TEMPLATE = """Based on the title '{title}', authors '{authors}', and abstract '{abstract}', please determine the scientific category of this paper. Additional info: {additional_info}\n{options_list}.
"""

def generate_template_library():
    """Generate a library of 20 templates (16 English, 4 Chinese) with varying field orders"""
    
    templates = []
    
    # English Templates (16)
    
    # 1. Standard Academic Style
    templates.append({
        "name": "Standard Academic",
        "language": "en",
        "template": """Please classify this research paper based on the following information:
Title: {title}
Authors: {authors}
Abstract: {abstract}
Additional details: {additional_info}

Select the most appropriate category from the options below:
{options_list}"""
    })
    
    # 2. Question Format
    templates.append({
        "name": "Question Format",
        "language": "en",
        "template": """Which scientific category best describes this paper?

Title: {title}
Authors: {authors}
Abstract: {abstract}
Additional information: {additional_info}

Choose from these options:
{options_list}"""
    })
    
    # 3. Instructional Style
    templates.append({
        "name": "Instructional",
        "language": "en",
        "template": """You are an expert in scientific paper classification. Examine the following paper details and select the correct category:

Abstract: {abstract}
Title: {title}
Authors: {authors}
Additional context: {additional_info}

Available categories:
{options_list}"""
    })
    
    # 4. Concise Format
    templates.append({
        "name": "Concise",
        "language": "en",
        "template": """Classify:
Authors: {authors}
Title: {title}
Abstract: {abstract}
Info: {additional_info}

Categories:
{options_list}"""
    })
    
    # 5. Journal Submission Style
    templates.append({
        "name": "Journal Submission",
        "language": "en",
        "template": """For journal submission purposes, please categorize this manuscript:

Additional info: {additional_info}
Title: {title}
Authors: {authors}
Abstract: {abstract}

Select one primary category:
{options_list}"""
    })
    
    # 6. Multiple Choice Exam Style
    templates.append({
        "name": "Multiple Choice",
        "language": "en",
        "template": """Question: What is the primary scientific category of this paper?

Authors: {authors}
Title: {title}
Abstract: {abstract}
Additional notes: {additional_info}

Options:
{options_list}"""
    })
    
    # 7. Research Assistant Style
    templates.append({
        "name": "Research Assistant",
        "language": "en",
        "template": """[Research Assistant Task] Categorize this paper:

Abstract: {abstract}
Notes: {additional_info}
Title: {title}
Authors: {authors}

Possible categories:
{options_list}"""
    })
    
    # 8. Bibliographic Style
    templates.append({
        "name": "Bibliographic",
        "language": "en",
        "template": """Bibliographic Classification Request:

Metadata: {additional_info}
Paper: {title} by {authors}
Abstract: {abstract}

Please assign to one of these categories:
{options_list}"""
    })
    
    # 9. Technical Report Style
    templates.append({
        "name": "Technical Report",
        "language": "en",
        "template": """TECHNICAL CLASSIFICATION REQUEST
================================
Supplementary Data: {additional_info}
Document Title: {title}
Authors: {authors}
Abstract: {abstract}

Classification Options:
{options_list}"""
    })
    
    # 10. Conversational Style
    templates.append({
        "name": "Conversational",
        "language": "en",
        "template": """Hey there! Could you help categorize this research paper?

Some extra details: {additional_info}
It's titled "{title}" by {authors}.
Here's the abstract: {abstract}

Which of these categories fits best?
{options_list}"""
    })
    
    # 11. Formal Request Style
    templates.append({
        "name": "Formal Request",
        "language": "en",
        "template": """Classification Request Form
---------------------------
Additional Information: {additional_info}
Paper Title: {title}
Author(s): {authors}
Abstract: {abstract}

Please indicate the most appropriate category:
{options_list}"""
    })
    
    # 12. Scientific Database Style
    templates.append({
        "name": "Scientific Database",
        "language": "en",
        "template": """SCIENTIFIC PAPER CATALOG ENTRY
Metadata: {additional_info}
Title: {title}
Authors: {authors}
Abstract: {abstract}

Required field: Primary Category
Available options:
{options_list}"""
    })
    
    # 13. Library Catalog Style
    templates.append({
        "name": "Library Catalog",
        "language": "en",
        "template": """LIBRARY CLASSIFICATION CARD
Notes: {additional_info}
Title: {title}
Author(s): {authors}
Abstract: {abstract}

Subject Classification:
{options_list}"""
    })
    
    # 14. AI Training Style
    templates.append({
        "name": "AI Training",
        "language": "en",
        "template": """[AI Training Example] Paper Classification Task:

Context: {additional_info}
Input:
Title: {title}
Authors: {authors}
Abstract: {abstract}

Correct output is one of:
{options_list}"""
    })
    
    # 15. Expert Review Style
    templates.append({
        "name": "Expert Review",
        "language": "en",
        "template": """Expert Classification Review
Review Notes: {additional_info}
Paper: {title}
Authors: {authors}
Abstract Summary: {abstract}

Please select primary discipline:
{options_list}"""
    })
    
    # 16. Minimalist Style
    templates.append({
        "name": "Minimalist",
        "language": "en",
        "template": """Classify this paper:
Notes: {additional_info}
"{title}" by {authors}
Abstract: {abstract}

Categories:
{options_list}"""
    })
    
    # Chinese Templates (4)
    
    # 17. 标准学术风格 (Standard Academic Style)
    templates.append({
        "name": "标准学术风格",
        "language": "zh",
        "template": """请根据以下信息对这篇研究论文进行分类：

标题: {title}
作者: {authors}
摘要: {abstract}
附加信息: {additional_info}

从以下选项中选择最合适的类别:
{options_list}"""
    })
    
    # 18. 问题格式 (Question Format)
    templates.append({
        "name": "问题格式",
        "language": "zh",
        "template": """以下论文最适合哪个科学类别？

附加信息: {additional_info}
标题: {title}
作者: {authors}
摘要: {abstract}

请从以下选项中选择:
{options_list}"""
    })
    
    # 19. 图书馆风格 (Library Style)
    templates.append({
        "name": "图书馆分类",
        "language": "zh",
        "template": """论文分类请求:

摘要: {abstract}
附加说明: {additional_info}
标题: {title}
作者: {authors}

请选择学科分类:
{options_list}"""
    })
    
    # 20. 简洁格式 (Concise Format)
    templates.append({
        "name": "简洁格式",
        "language": "zh",
        "template": """论文分类:
作者: {authors}
附加信息: {additional_info}
标题: {title}
摘要: {abstract}

可选类别:
{options_list}"""
    })
    
    return templates

OPTION_LIST = ""
# 动态生成分类选项，这里直接使用 LABEL_MAP
for label, code in LABEL_MAP.items():
    OPTION_LIST += f"{code}. {label}\n"
OPTION_LIST += "\n" # 添加一个空行


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


def process_data(example, template_library, base_template_ratio=0.6):
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

    if random.random() <= base_template_ratio:
        sampled_template = PROMPT_TEMPLATE
    else:
        sampled_template = random.choice(template_library)["template"]

    try:
        prompt = sampled_template.format(title=title, authors=authors, abstract=abstract,
                                         additional_info=additional_info, options_list=OPTION_LIST)
    except KeyError as e:
        print(f"Error formatting prompt: Missing key {e} in example: {example}")
        return None

    if category not in LABEL_MAP:
        print(f"Error: Category '{category}' not found in LABEL_MAP.")
        return None

    example["human"] = prompt
    example["assistant"] = LABEL_MAP[category]

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
            # print(f"Skipping item due to missing input or output: {item}")
            pass
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
    base_template_ratio = 0.6

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


    # 3 采样模板
    template_library = generate_template_library()
    print("sampled_ds", len(sampled_ds))
    # 4. Process the data to create input and output prompts
    print("Processing sampled data to create prompts...")
    sampled_ds = sampled_ds.map(
        lambda example: process_data(example, template_library, base_template_ratio),
        num_proc=NUM_PROCESSES,
        batched=False
    )
    print("sampled_ds22", len(sampled_ds))
    print("过滤空数据...")
    sampled_ds = sampled_ds.filter(lambda x: x is not None)

    # 保存采样的数据
    print("保存采样数据...")
    print("sampled_ds22", len(sampled_ds))
    print(sampled_ds)
    sampled_ds_chat = create_conversation_dataset(sampled_ds, SYSTEM_PROMPT)
    print("sampled_ds_chat",sampled_ds_chat)
    save_dataset_to_jsonl(sampled_ds_chat, SAMPLED_OUTPUT_FILE, FORCE_ASCII)

    print("\n--- Data processing pipeline completed. ---")


if __name__ == "__main__":
    main()
