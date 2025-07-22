'''
@author:zjh
@date: 25/07/21
@desc: 在多类采样的基础上，添加更多模板，取90%比例原始模板，10%概率取多样的模板。（注意：系统提示词也要同步改）

待修改:
1 PROMPT_TEMPLATE √
2 generate_prompt√
3 process_data
4 参数0.9√
5 run函数
6 OPTION_LIST
7 system也要修改
'''
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


# --- Configuration Parameters ---
DATA_FILE = "../data_arxiv/arxiv-metadata-oai-snapshot.json"
# DATA_FILE = "../data_arxiv/corpus.jsonl"


# 获取类别到选项的映射
LABEL_MAP = extract_category_mapping()
NEW_LABELS = list(LABEL_MAP.keys())


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
NUM_PROCESSES = 16 # 根据你的CPU核心数调整

FORCE_ASCII = False  # Set to True if ASCII encoding is required



HANDLE_MULTI_LABELS = True
SAMPLES_PER_LABEL = 600
FILTERED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.9/filtered_data.jsonl"
SAMPLED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.9/sample_data_600.jsonl"
PROMPT_DIVERSE_RATIO = 0.9 # 模板切换的比例，90%默认的模板，10%使用多样的模板

# FILTERED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.7/filtered_data.jsonl"
# SAMPLED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.7/sample_data_600.jsonl"
# PROMPT_DIVERSE_RATIO = 0.7

FILTERED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.5/filtered_data.jsonl"
SAMPLED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.5/sample_data_600.jsonl"
PROMPT_DIVERSE_RATIO = 0.5 

# FILTERED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.3/filtered_data.jsonl"
# SAMPLED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.3/sample_data_600.jsonl"
# PROMPT_DIVERSE_RATIO = 0.3 

FILTERED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.1/filtered_data.jsonl"
SAMPLED_OUTPUT_FILE = "data/mclass_mprompt/ratio0.1/sample_data_600.jsonl"
PROMPT_DIVERSE_RATIO = 0.1 





###################### 修改 ######################
PROMPT_TEMPLATE = """Based on the title '{title}', authors '{authors}', and abstract '{abstract}', please determine the scientific category of this paper. Additional info: {additional_info}\n{options_list}.
"""

OPTION_LIST = ""
# 动态生成分类选项，这里直接使用 LABEL_MAP
for label, code in LABEL_MAP.items():
    OPTION_LIST += f"{code}. {label}\n"
OPTION_LIST += "\n" # 添加一个空行

SYSTEM_PROMPTS = [
    # --- 英文指令 (约16个，80%比例) ---
    # 长指令 (约8-10个)
    "You are a premier expert in scientific classification, possessing deep interdisciplinary knowledge. Based on the paper's title, authors, abstract, and provided additional information, accurately determine its scientific category. Focus on providing the most relevant single category code.",
    "Act as an efficient academic data analyst. Carefully review the paper's metadata and additional descriptions, extract key information, and select the most fitting category option. Ensure your response is a direct classification code.",
    "Assume the role of a rigorous academic reviewer, performing initial subject classification for submitted papers. Ensure your judgment is based on all provided information and strictly adheres to the classification criteria. Provide only the category code.",
    "You are an AI trained for academic paper indexing. Analyze the given title, authors, abstract, and any extra notes to assign the correct scientific discipline. Your output should be a single, valid category.",
    "As a research librarian, you help organize academic literature. Classify this paper by selecting the most appropriate subject from the given options. Your final answer must be just the category code.",
    "You are an intelligent agent specialized in scientific document categorization. Focus on identifying the core research area from the provided inputs, and then output the corresponding category code.",
    "You are a methodical classifier. Carefully evaluate the paper's title, authors, abstract, and supplementary data before assigning a category. Respond only with the category code.",
    "Your mission is to classify academic papers accurately. Pay close attention to the abstract and any supplementary details to make your selection. Provide the category code.",
    "You are an automated cataloging system for scientific research. Assign the correct category to the paper based on the input data. Output only the category.",

    # 短指令 (约6-8个)
    "Classify this paper by category.",
    "Provide the scientific category for this paper.",
    "Determine the paper's main research category.",
    "Select the best category from the options.",
    "Categorize the provided academic paper.",
    "Identify the scientific discipline.",
    "Assign a category to this paper.",
    "What is the paper's category?",

    # --- 中文指令 (约4个，20%比例) ---
    # 长指令 (约2-3个)
    "你是一位顶尖的学术论文分类专家，拥有深厚的跨学科知识。请依据论文的标题、作者和摘要，以及提供的附加信息，准确判断其所属的科学类别。请仅输出类别代码。",
    "你是一个高效的数据分析助手，擅长处理学术信息。你需要仔细审阅论文的元数据和附加描述，从中提取关键信息，并选择最贴切的分类选项。最终请直接输出分类代码。",
    "你扮演一位严谨的学术审核员，对提交的论文进行初步的学科分类。请确保你的判断基于所有提供的信息，并严格遵循提供的分类标准。只输出分类的字母代码。",

    # 短指令 (约1-2个)
    "请为这篇论文分类。",
    "论文分类：给出最合适的类别代码。"
]
##################################################


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


### 修改：更多模板
def process_data(example, template_library, prompt_diverse_ratio=0.6):
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

    if random.random() <= prompt_diverse_ratio:
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



def create_conversation_dataset(dataset, prompt_diverse_ratio=0.9):
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
            # TODO: 随机采样系统提示词
            if random.random() <= prompt_diverse_ratio:
                system_prompt = SYSTEM_PROMPT
            else:
                system_prompt = random.choice(SYSTEM_PROMPTS)

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

    # 3 采样模板
    template_library = generate_template_library()
    print("sampled_ds", len(sampled_ds))

    
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
        lambda example: process_data(example, template_library, PROMPT_DIVERSE_RATIO),
        num_proc=NUM_PROCESSES,
        batched=False
    )

    print("过滤空数据...")
    sampled_ds = sampled_ds.filter(lambda x: x is not None)

    # 保存采样的数据
    print("保存采样数据...")
    sampled_ds_chat = create_conversation_dataset(sampled_ds, prompt_diverse_ratio=PROMPT_DIVERSE_RATIO)
    save_dataset_to_jsonl(sampled_ds_chat, SAMPLED_OUTPUT_FILE, FORCE_ASCII)

    # 保存过滤的数据
    FILTER_CHAT_FILE = FILTERED_OUTPUT_FILE.replace(".jsonl","_chat.jsonl")
    filtered_ds_chat = filtered_ds.map(
        lambda example: process_data(example, template_library, PROMPT_DIVERSE_RATIO),
        num_proc=NUM_PROCESSES,
        batched=False
    )
    print("filtered_ds_chat 1 ", filtered_ds_chat)
    filtered_ds_chat =  create_conversation_dataset(filtered_ds_chat, prompt_diverse_ratio=PROMPT_DIVERSE_RATIO)
    print("filtered_ds_chat 2 ", filtered_ds_chat)

    save_dataset_to_jsonl(filtered_ds_chat, FILTER_CHAT_FILE, FORCE_ASCII)

    print("\n--- Data processing pipeline completed. ---")


if __name__ == "__main__":
    main()
