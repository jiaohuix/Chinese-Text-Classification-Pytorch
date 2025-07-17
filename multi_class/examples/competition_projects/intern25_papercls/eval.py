import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any
from swift.llm import PtEngine, RequestConfig, InferRequest
from datasets import load_dataset
from tqdm import tqdm

# --- 配置和日志设置 ---

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # 默认输出到控制台

# 设置 CUDA 可见设备，如果需要从命令行覆盖，将在 argparse 中处理
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 此处暂时注释掉，将由命令行参数控制

# --- 辅助函数 ---

def format_request(prompt: str) -> InferRequest:
    """
    将用户输入的 prompt 格式化为 InferRequest 对象。
    系统消息固定为“你是个优秀的论文分类师”。
    """
    messages = [
        {'role': 'system', 'content': '你是个优秀的论文分类师'},
        {'role': 'user', 'content': prompt},
    ]
    return InferRequest(messages)

def extract_prediction_label(prediction_text: str, original_label: str) -> str:
    """
    从模型预测的文本中提取用于比对的标签（通常是第一个字母）。
    如果模型输出是完整的句子，也尝试提取第一个字母。
    """
    prediction_text = prediction_text.strip()
    if not prediction_text:
        return ""

    # 尝试直接从开头提取字母 A, B, C, D 等
    for char_code in range(ord('A'), ord('Z') + 1):
        char = chr(char_code)
        if prediction_text.startswith(char):
            return char

    # 如果不是以字母开头，尝试提取第一个字母字符
    return prediction_text[0]

def save_results(output_dir: str, metrics: Dict[str, Any], predictions: List[Dict[str, Any]], bad_cases: List[Dict[str, Any]]):
    """
    将评估结果保存到指定目录下的不同文件中。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存 metrics
    metric_file = os.path.join(output_dir, 'metric.json')
    try:
        with open(metric_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        logging.info(f"✔️ 评估指标已保存到: {metric_file}")
    except IOError as e:
        logging.error(f"写入指标文件 {metric_file} 时出错: {e}")

    # 保存所有预测结果
    prediction_file = os.path.join(output_dir, 'prediction.jsonl')
    try:
        with open(prediction_file, 'w', encoding='utf-8') as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"✔️ 所有预测结果已保存到: {prediction_file}")
    except IOError as e:
        logging.error(f"写入预测文件 {prediction_file} 时出错: {e}")

    # 保存错误案例
    bad_case_file = os.path.join(output_dir, 'bad_case.jsonl')
    try:
        with open(bad_case_file, 'w', encoding='utf-8') as f:
            for item in bad_cases:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"✔️ 错误案例已保存到: {bad_case_file}")
    except IOError as e:
        logging.error(f"写入错误案例文件 {bad_case_file} 时出错: {e}")

# --- 核心评估函数 ---

def evaluate_model(model_path: str, dataset_path: str, batch_size: int, output_dir: str):
    """
    加载模型，加载数据集，进行批量推理，评估准确率，并记录结果。

    Args:
        model_path (str): LLM 模型权重路径。
        dataset_path (str): 评估数据集文件路径 (JSON Lines 格式)。
        batch_size (int): 推理的批量大小。
        output_dir (str): 保存评估结果的目录。
    """
    logging.info("🚀 开始 LLM 模型评估...")
    start_time = time.time()

    # --- 1. 加载模型 ---
    logging.info(f"正在加载模型: {model_path}")
    try:
        # 设置推理配置
        # max_batch_size 需要在 PtEngine 初始化时指定
        request_config = RequestConfig(max_tokens=2048, temperature=0)
        engine = PtEngine(model_path, max_batch_size=batch_size)
        logging.info("✔️ 模型加载成功。")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    # --- 2. 加载数据集 ---
    logging.info(f"正在加载数据集: {dataset_path}")
    try:
        # load_dataset 默认从 huggingface hub 加载，如果本地文件需要指定 data_files
        evaluation_data = load_dataset('json', data_files=dataset_path, split="train")
        total_samples = len(evaluation_data)
        logging.info(f"✔️ 成功加载 {total_samples} 个样本。")
        if total_samples == 0:
            logging.warning("数据集为空，无法进行评估。")
            return
    except FileNotFoundError:
        logging.error(f"数据集文件未找到: {dataset_path}")
        return
    except Exception as e:
        logging.error(f"加载数据集 {dataset_path} 时发生错误: {e}")
        logging.error("请确保数据集文件存在且格式正确 (JSON Lines)，且每行包含 'text' 和 'text_label' 字段。")
        return

    # --- 3. 进行批量推理和评估 ---
    logging.info(f"开始进行批量推理 (batch_size={batch_size})...")
    all_results = []
    correct_predictions = 0

    # 使用 tqdm 包装迭代器，显示进度条
    for i in tqdm(range(0, total_samples, batch_size), desc="评估进度", unit="batch"):
        batch_data = evaluation_data[i : i + batch_size]
        texts_in_batch = batch_data["text"]
        labels_in_batch = batch_data["text_label"]

        # 构建批次内的 InferRequest 列表
        batch_infer_requests = [format_request(text) for text in texts_in_batch]

        try:
            # 进行推理
            resp_list = engine.infer(batch_infer_requests, request_config)

            # 处理每个样本的推理结果
            for j, (text, resp, original_label) in enumerate(zip(texts_in_batch, resp_list, labels_in_batch)):
                model_output = resp.choices[0].message.content.strip()
                # extracted_label = extract_prediction_label(model_output, original_label)
                extracted_label = model_output[:1]
                
                is_correct = extracted_label.upper() == original_label.upper()

                # 记录单个样本结果
                sample_result = {
                    "input_text": text,
                    "label": original_label,
                    "prediction": model_output,
                    "extracted_prediction": extracted_label,
                    "is_correct": is_correct
                }
                all_results.append(sample_result)

                if is_correct:
                    correct_predictions += 1

        except Exception as e:
            logging.error(f"处理批次 {i//batch_size + 1} 时发生推理错误: {e}")
            # 记录下批次中的错误，但不中断整个评估
            for j, text in enumerate(texts_in_batch):
                original_label = labels_in_batch[j]
                all_results.append({
                    "input_text": text,
                    "label": original_label,
                    "prediction": f"ERROR: {e}",
                    "extracted_prediction": "",
                    "is_correct": False
                })

    # --- 4. 计算和保存评估指标 ---
    end_time = time.time()
    total_duration = end_time - start_time
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    # 计算 QPS (Queries Per Second)
    # 注意：这里的 QPS 计算是基于 batch_size，而不是单个请求
    # 如果想计算 tokens/sec, 需要从 resp_list 中获取 token 数量
    # 这里简化为 "completed_batches / duration"
    completed_batches = (total_samples + batch_size - 1) // batch_size
    qps = completed_batches / total_duration if total_duration > 0 else 0

    metrics = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "accuracy": round(accuracy, 2),
        "qps": round(qps, 2), # 以 batch 为单位的 QPS
        "total_duration_seconds": round(total_duration, 2)
    }

    # --- 5. 整理 bad_case 数据 ---
    bad_cases = [res for res in all_results if not res["is_correct"]]

    # --- 6. 保存所有结果 ---
    save_results(output_dir, metrics, all_results, bad_cases)

    # --- 7. 打印最终摘要 ---
    logging.info("\n" + "="*30)
    logging.info("🌟 评估摘要 🌟")
    logging.info("="*30)
    logging.info(f"模型路径: {model_path}")
    logging.info(f"数据集: {dataset_path}")
    logging.info(f"总样本数: {total_samples}")
    logging.info(f"正确预测数: {correct_predictions}")
    logging.info(f"准确率: {accuracy:.2f}%")
    logging.info(f"总耗时: {total_duration:.2f} 秒")
    logging.info(f"吞吐量 (batch/sec): {qps:.2f}")
    logging.info(f"结果已保存到目录: {output_dir}")
    logging.info("="*30)


# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 模型评估脚本")
    parser.add_argument(
        "-i", "--input_dataset",
        type=str,
        required=True,
        help="评估数据集的路径 (JSON Lines 格式，包含 'text' 和 'text_label' 字段)。"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="要评估的 LLM 模型权重路径。"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./evaluation_output",
        help="保存评估结果 (metric.json, prediction.jsonl, bad_case.jsonl) 的目录。"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=4,
        help="推理的批量大小 (batch size)。"
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0",
        help="设置 CUDA_VISIBLE_DEVICES 环境变量，指定使用的 GPU ID (e.g., '0' or '0,1')。"
    )

    args = parser.parse_args()

    # 设置 CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    logging.info(f"设置 CUDA_VISIBLE_DEVICES 为: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # 确保 Swift LLM 的依赖已安装
    try:
        from swift.llm import PtEngine
    except ImportError:
        logging.error("错误: 'swift-llm' 库未安装。请运行 'pip install swift-llm' 进行安装。")
        exit()
    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("错误: 'datasets' 库未安装。请运行 'pip install datasets' 进行安装。")
        exit()
    try:
        from tqdm import tqdm
    except ImportError:
        logging.error("错误: 'tqdm' 库未安装。请运行 'pip install tqdm' 进行安装。")
        exit()

    # 检查数据集文件是否存在
    if not os.path.exists(args.input_dataset):
        logging.error(f"错误: 数据集文件 '{args.input_dataset}' 不存在。请提供有效路径。")
    else:
        evaluate_model(args.model_path, args.input_dataset, args.batch_size, args.output_dir)
