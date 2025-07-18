# inference_onnx.py

'''
@author: jiaohuix (adapted for ONNX by AI)
@file: inference_onnx.py
@date: 2025/07/17 14:39


# 示例 ONNX 模型推理 (不需要 label_file, 也不需要输入文件有 label 列)
python inference_onnx.py \
    --input data/test.jsonl \
    --output_dir results/onnx_prediction_only \
    --model_dir path/to/your/onnx_model_dir \
    --text_column "text" \
    --onnx_file "model.onnx" # 或 "model_quantized.onnx"


# 示例 ONNX 模型推理并尝试转换为文本标签 (需要 label_file)
python inference_onnx.py \
    --input data/test.jsonl \
    --output_dir results/onnx_prediction_with_mapping \
    --model_dir path/to/your/onnx_model_dir \
    --label_file path/to/labels.json \
    --text_column "text" \
    --onnx_file "model.onnx"


# 示例 ONNX 模型评估模式 (需要 label_file 和输入文件包含 label_column)
python inference_onnx.py \
    --input data/test_with_labels.jsonl \
    --output_dir results/onnx_evaluation_run \
    --model_dir path/to/your/onnx_model_dir \
    --label_file path/to/labels.json \
    --text_column "text" \
    --label_column "text_label" \
    --do_eval \
    --batch_size 32 \
    --onnx_file "model.onnx" # 或 "model_quantized.onnx"


# 简短参数示例
python inference_onnx.py -i data/test.jsonl -o results_onnx -m ckpt/onnx_model_dir -l ckpt/labels.json --do_eval --batch_size 32 --onnx_file "model.onnx"


python inference.py -i data/patient_v252/test.jsonl -o results/test_onnx/fp32 -m ckpt/patient_intention/v3_sensitive2k_zhuanbing/baseline/ --do_eval -l data/labels.json --device -1
python inference_onnx.py -i data/patient_v252/test.jsonl -o  results/test_onnx/fp32_onnx -m ckpt/patient_intention/v3_sensitive2k_zhuanbing/baseline/onnx --do_eval -l data/labels.json    --onnx_file "model.onnx" --device -1
python inference_onnx.py -i data/patient_v252/test.jsonl -o  results/test_onnx/int8_onnx -m ckpt/patient_intention/v3_sensitive2k_zhuanbing/baseline/onnx_quantized --do_eval -l data/labels.json    --onnx_file "model_quantized.onnx" --device -1

| 模型类型       | 样本数量 | 准确率  | 错误案例数 | 总执行时间（秒） | QPS   |
|----------------|----------|---------|------------|------------------|-------|
| FP32（原模型） | 2002     | 0.8531  | 294        | 14.5618          | 137.48|
| FP32（ONNX）   | 2002     | 0.8531  | 294        | 13.8005          | 145.07|
| INT8（ONNX量化）| 2002     | 0.7622  | 476        | 10.7145          | 186.85|
'''

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path # For cleaner path handling
from typing import List
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer # Only need tokenizer here
# Import ONNX runtime related classes from optimum
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.utils import logging as optimum_logging

# Set Optimum logging level to error to avoid verbose output from ONNX session creation
optimum_logging.set_verbosity_error()


# --- Helper Functions (mostly same as original, with ONNX considerations) ---

def load_label_map(label_file):
    """
    加载标签映射文件 (JSON 格式)。

    参数:
        label_file (str): 包含标签映射的 JSON 文件路径。
                          例如: {"类别文本1": 0, "类别文本2": 1, ...}

    返回:
        dict: 包含数字标签到文本标签的映射字典。
    """
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        # 构建数字标签到文本标签的映射
        # Ensure keys are integers for mapping lookup
        id_to_label_map = {int(v): k for k, v in label_data.items()}
        print(f"🏷️ 成功加载标签映射文件: {label_file}")
        print(f"   标签映射 (数字 -> 文本): {id_to_label_map}")
        return id_to_label_map
    except FileNotFoundError:
        print(f"❌ 错误: 标签文件未找到: {label_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ 错误: 无法解析 JSON 文件: {label_file}")
        sys.exit(1)
    except ValueError: # 如果数字键转换失败
        print(f"❌ 错误: JSON 文件中的标签值必须是数字（整数）。请检查 {label_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载标签映射时发生未知错误: {e}")
        sys.exit(1)

def load_data(input_file, text_column='text', label_column='text_label'):
    """
    加载输入数据。

    参数:
        input_file (str): 输入文件路径 (CSV, TSV, JSON 或 JSONL 格式)。
        text_column (str): 包含文本数据的列名。
        label_column (str): 包含实际标签（文本形式）的列名。

    返回:
        tuple: 包含文本列表、实际文本标签列表 (如果存在) 和数据长度的元组。
    """
    print(f"📦 正在加载数据: {input_file}")
    try:
        # Hugging Face datasets load_dataset is generally robust for these formats
        # We explicitly handle JSONL for better compatibility
        if input_file.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=input_file, split="train", streaming=False)
        elif input_file.endswith('.json'):
            dataset = load_dataset("json", data_files=input_file, split="train", streaming=False)
        elif input_file.endswith('.csv'):
            dataset = load_dataset("csv", data_files=input_file, split="train", streaming=False)
        elif input_file.endswith('.tsv'):
            # For TSV, we can read it as CSV with a different separator
            dataset = load_dataset("csv", data_files=input_file, split="train", streaming=False, separator='\t')
        else:
            raise ValueError("不支持的文件格式。请提供 CSV, TSV, JSON 或 JSONL 文件。")

        df = dataset.to_pandas()

        if text_column not in df.columns:
            raise ValueError(f"指定的文本列 '{text_column}' 不存在于文件中。可用列: {df.columns.tolist()}")

        text_ls = df[text_column].tolist()
        data_length = len(text_ls)
        print(f"✅ 成功加载数据，共 {data_length} 条样本")
        print(f"📊 数据形状: {df.shape}")
        print(f"✏️ 用于文本的列名: '{text_column}'")

        actual_labels_text_ls = None
        label_column_present = False
        if label_column in df.columns:
            actual_labels_text_ls = df[label_column].tolist()
            label_column_present = True
            print(f"🏷️ 检测到实际标签列: '{label_column}' (包含文本标签)")
        else:
            print(f"⚠️ 未在文件中找到指定的标签列: '{label_column}'。如果需要评估，请确保此列存在。")

        return text_ls, actual_labels_text_ls, data_length, label_column_present

    except FileNotFoundError:
        print(f"❌ 错误: 输入文件未找到: {input_file}")
        sys.exit(1)
    except ValueError as ve:
        print(f"❌ 数据加载错误: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载数据时发生未知错误: {e}")
        sys.exit(1)


def load_model_and_classifier(model_dir, onnx_file, device, max_length, batch_size):
    """
    加载 ONNX 模型、tokenizer 并创建 inference pipeline。
    假设模型输出的是数字标签 ID。

    参数:
        model_dir (str): 包含 ONNX 模型和 tokenizer 的目录。
        onnx_file (str): ONNX 模型文件名 (e.g., "model.onnx", "model_quantized.onnx").
        device (int): 使用的 GPU 设备 ID (0, 1, ...). -1 for CPU.
        max_length (int): Transformer 模型处理的最大输入长度。
        batch_size (int): 批处理大小。

    返回:
        tuple: (tokenizer, model_runner)
               tokenizer: Hugging Face tokenizer.
               model_runner: A callable that takes texts and returns raw model outputs.
    """
    print(f"🧠 正在加载 ONNX 模型: {model_dir} (文件: {onnx_file})")
    try:
        # Load tokenizer from the model directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load ONNX model using ORTModelForSequenceClassification
        # The model_dir should be the directory containing the ONNX files and tokenizer files.
        # ORTModelForSequenceClassification handles the execution providers based on device.
        # If device is -1 (CPU), it uses CPUExecutionProvider. If device >= 0, it tries CUDAExecutionProvider.
        # We need to specify the file_name here.
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_dir,
            file_name=onnx_file,
            # ONNX Runtime allows specifying providers and their order.
            # For CUDA: ["CUDAExecutionProvider", "CPUExecutionProvider"]
            # For CPU: ["CPUExecutionProvider"]
            # Optimum handles this based on the device argument.
            # We pass the device argument, which optimum interprets.
            provider_to_id_map={
                "CPUExecutionProvider": 0,
                "CUDAExecutionProvider": 1
            },
            provider_options={
                "CPUExecutionProvider": {},
                "CUDAExecutionProvider": {"device_id": str(device)} if device >= 0 else {}
            },
            # This ensures that the model is loaded with the correct device context (for ONNX runtime)
            # For ORTModel, this might refer to the desired execution provider.
            # We pass the device argument to from_pretrained and let optimum handle it.
            # If you want to explicitly control, you might need to pass providers manually.
            # For simplicity, we rely on optimum's interpretation of the device argument.
        )

        # The ORTModel object itself is callable for inference.
        # We wrap it for consistency with how transformers pipeline might be used.
        def model_runner(texts: List[str]):
            # Tokenize inputs for ONNX runtime. Use "pt" for PyTorch tensors, which ORTModel can handle.
            # ORTModel might also support "np" directly. Let's stick to "pt" for now.
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt" # Use PyTorch tensors
            )
            # ORTModel expects a dictionary of tensors.
            # It will handle moving to the correct provider internally.
            outputs = ort_model(**inputs)
            # ORTModelForSequenceClassification typically returns a BaseModelOutput object
            # with 'logits' attribute.
            return outputs.logits

        print(f"✅ ONNX 模型和 Tokenizer 加载成功。设备: {'GPU:' + str(device) if device >= 0 else 'CPU'}, 最大长度: {max_length}, 批处理大小: {batch_size}")
        return tokenizer, model_runner

    except FileNotFoundError:
        print(f"❌ 错误: 模型文件或 tokenizer 文件未在 '{model_dir}' 中找到。请确保 '{onnx_file}' 存在。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载 ONNX 模型或 tokenizer 时发生错误: {e}")
        sys.exit(1)



def batch_predict(model_runner, text_ls, batch_size=16):
    """
    使用 ONNX 模型运行程序进行批量预测，并显示进度条。

    参数:
        model_runner: 接受文本列表并返回原始模型输出的可调用对象。
        text_ls (list): 文本列表。
        batch_size (int): 批量大小。

    返回:
        tuple: 预测结果列表，开始时间和结束时间。
    """
    print("🚀 开始进行 ONNX 模型预测...")
    start_time = time.time()
    all_batch_outputs = [] # Store outputs from each batch as they are
    num_samples = len(text_ls)

    effective_batch_size = min(batch_size, num_samples)
    if effective_batch_size <= 0:
        print("⚠️ 输入数据为空，返回空输出。")
        return np.empty((0, 0), dtype=np.float32), start_time, time.time()

    for i in tqdm(range(0, num_samples, effective_batch_size), desc="推理进度"):
        batch_texts = text_ls[i : i + effective_batch_size]
        batch_output_logits = None # Initialize to None
        try:
            # model_runner returns something that should be processed as logits.
            # Based on the log, it's a torch.Tensor.
            raw_output = model_runner(batch_texts)

            # --- NEW: Convert torch.Tensor to numpy.ndarray ---
            if isinstance(raw_output, torch.Tensor):
                batch_output_logits = raw_output.detach().cpu().numpy() # Convert to numpy
            elif isinstance(raw_output, np.ndarray) and raw_output.ndim == 2:
                batch_output_logits = raw_output # It's already a numpy array
            # --- END NEW ---

            else:
                # Got something, but not the expected format
                print(f"\n❌ 批次 {i//effective_batch_size + 1}: 预测输出格式不正确。")
                print(f"   - 类型: {type(raw_output)}")
                print(f"   - 维度: {getattr(raw_output, 'ndim', 'N/A')}")
                if isinstance(raw_output, np.ndarray):
                    print(f"   - 形状: {raw_output.shape}")
                # Skip this batch if it's not in the expected format

            # After potential conversion, check if we have valid logits
            if batch_output_logits is not None:
                if isinstance(batch_output_logits, np.ndarray) and batch_output_logits.ndim == 2:
                    all_batch_outputs.append(batch_output_logits)
                else:
                    # This case should ideally not happen after conversion, but as a safeguard
                    print(f"\n❌ 批次 {i//effective_batch_size + 1}: 转换后格式仍不正确。")
                    print(f"   - 类型: {type(batch_output_logits)}")
                    print(f"   - 维度: {getattr(batch_output_logits, 'ndim', 'N/A')}")
                    if isinstance(batch_output_logits, np.ndarray):
                        print(f"   - 形状: {batch_output_logits.shape}")


        except Exception as e:
            # Caught an exception during model_runner execution
            print(f"\n❌ 预测批次 {i//effective_batch_size + 1} 时发生错误: {e}")
            # import traceback # Uncomment for detailed stack trace
            # traceback.print_exc()

    end_time = time.time()
    print("✅ ONNX 模型预测完成。")

    if not all_batch_outputs:
        print("❌ 警告: 没有成功的批次输出。")
        return np.empty((0, 0), dtype=np.float32), start_time, end_time

    try:
        concatenated_outputs = np.concatenate(all_batch_outputs, axis=0)
    except ValueError as e:
        print(f"❌ 错误: 无法拼接批次输出。可能的形状不一致：{e}")
        print("   尝试打印部分批次形状以帮助调试：")
        for idx, arr in enumerate(all_batch_outputs[:5]):
            print(f"     Batch {idx}: shape={arr.shape}, dtype={arr.dtype}")
        concatenated_outputs = np.empty(len(all_batch_outputs), dtype=object)
        for idx, arr in enumerate(all_batch_outputs):
            concatenated_outputs[idx] = arr
        print("   创建了 object 类型的数组作为回退。")

    return concatenated_outputs, start_time, end_time


def convert_prediction_to_text(outputs, id_to_label_map):
    """
    将模型的数字 ID 预测转换为文本标签，使用提供的映射。
    ONNX 模型通常输出 NumPy 数组。

    参数:
        outputs (np.ndarray): 原始预测输出 (logits numpy array, shape [num_samples, num_labels]).
        id_to_label_map (dict): 数字标签 ID 到文本标签的映射。

    返回:
        list: 包含转换后的文本标签和分数的列表。
              例如: [{'prediction_text': '类别文本', 'score': 0.9}, ...]
    """
    converted_predictions = []
    if not id_to_label_map:
        print("⚠️ 未提供标签映射，无法将预测 ID 转换为文本。将保留数字 ID。")
        # If no map, return raw indices as string.
        for i, sample_logits in enumerate(outputs):
            predicted_id = np.argmax(sample_logits)
            converted_predictions.append({
                'prediction_text': str(predicted_id),
                'score': float(np.max(sample_logits)) # Use max logit value as score
            })
        return converted_predictions

    # Apply softmax to logits to get probabilities
    # Add small epsilon to avoid division by zero in softmax
    exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probabilities = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-9)

    # Get the predicted index for each sample
    predicted_indices = np.argmax(outputs, axis=1)

    for i, pred_idx in enumerate(predicted_indices):
        try:
            # Use the predicted index to get the text label from the map
            pred_text = id_to_label_map.get(pred_idx)

            if pred_text is None:
                print(f"⚠️ 警告: 预测的类 ID '{pred_idx}' 在标签映射中未找到。将使用占位符 '未知类别'。")
                pred_text = "未知类别"

            # Get the corresponding probability (confidence score)
            score = probabilities[i, pred_idx]

            converted_predictions.append({
                'prediction_text': pred_text,
                'score': float(score)
            })
        except IndexError:
            # If pred_idx is out of bounds for probabilities array
            print(f"⚠️ 警告: 预测索引 {pred_idx} 超出范围。")
            converted_predictions.append({
                'prediction_text': "索引错误",
                'score': 0.0
            })
        except Exception as e:
            print(f"❌ 转换预测时发生错误: {e}")
            converted_predictions.append({
                'prediction_text': "转换错误",
                'score': 0.0
            })
    return converted_predictions


def calculate_accuracy(predictions_text, actual_labels_text):
    """
    计算准确率，输入为文本标签列表。

    参数:
        predictions_text (list): 预测的文本标签列表。
        actual_labels_text (list): 实际的文本标签列表。

    返回:
        float: 准确率。
    """
    if not actual_labels_text:
        print("⚠️ 没有提供实际标签，无法计算准确率。")
        return 0.0

    if len(predictions_text) != len(actual_labels_text):
        print(f"❌ 错误: 预测文本数量 ({len(predictions_text)}) 与实际标签数量 ({len(actual_labels_text)}) 不匹配。")
        return 0.0

    correct_predictions = sum(
        pred == actual for pred, actual in zip(predictions_text, actual_labels_text)
    )
    total_predictions = len(actual_labels_text)

    if total_predictions == 0:
        return 0.0

    accuracy = correct_predictions / total_predictions
    return accuracy


def format_and_save_predictions(
    outputs, text_ls, actual_labels_text_ls, id_to_label_map, output_dir
):
    """
    格式化预测结果并保存到 prediction.jsonl 文件。

    参数:
        outputs (np.ndarray): ONNX 模型原始预测输出 (logits numpy array).
        text_ls (list): 输入的文本列表。
        actual_labels_text_ls (list): 实际的文本标签列表。
        id_to_label_map (dict): 数字标签 ID 到文本标签的映射。
        output_dir (str): 保存结果的目录。

    返回:
        list: 格式化后的预测结果列表，用于评估。
              每个元素是一个字典: {'text': ..., 'text_label': ..., 'prediction': ..., 'score': ...}
    """
    print("💾 正在格式化预测结果并准备保存...")

    # Convert the ONNX model outputs (logits) to text predictions and scores
    converted_predictions_data = convert_prediction_to_text(outputs, id_to_label_map)

    formatted_predictions_for_eval = []
    predictions_text_for_eval = [] # Only for calculating accuracy

    for i, text in enumerate(text_ls):
        # Ensure we don't go out of bounds for converted_predictions_data
        if i >= len(converted_predictions_data):
            print(f"⚠️ 警告: 预测数据不足，样本 {i} 无法处理。")
            continue

        pred_data = converted_predictions_data[i]
        predicted_label_text = pred_data['prediction_text']
        score = pred_data['score']

        # Get actual label text
        actual_label_text = None
        if actual_labels_text_ls and i < len(actual_labels_text_ls):
            actual_label_text = actual_labels_text_ls[i]

        formatted_predictions_for_eval.append({
            "text": text,
            "text_label": actual_label_text, # Save original actual label text
            "prediction": predicted_label_text, # Save converted predicted text
            "score": score
        })
        predictions_text_for_eval.append(predicted_label_text)

    # Save prediction.jsonl
    output_file_jsonl = os.path.join(output_dir, "prediction.jsonl")
    try:
        with open(output_file_jsonl, 'w', encoding='utf-8') as f:
            for entry in formatted_predictions_for_eval:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"📄 预测结果已保存到: {output_file_jsonl}")
    except Exception as e:
        print(f"❌ 保存 prediction.jsonl 时发生错误: {e}")

    return formatted_predictions_for_eval, predictions_text_for_eval


def evaluate_and_save_metrics(formatted_predictions_for_eval, predictions_text_for_eval, output_dir):
    """
    根据预测结果计算评估指标并保存 metrics.json 和 bad_cases.jsonl。

    参数:
        formatted_predictions_for_eval (list): 格式化后的预测结果列表。
        predictions_text_for_eval (list): 仅包含预测文本标签的列表，用于计算准确率。
        output_dir (str): 保存结果的目录。
    """
    if not formatted_predictions_for_eval:
        print("⚠️ 没有预测结果，无法进行评估。")
        return

    print("📊 开始评估模型性能...")

    actual_labels_text = [item['text_label'] for item in formatted_predictions_for_eval if item['text_label'] is not None]
    # Filter predictions to only include those with actual labels for evaluation
    valid_predictions_text = [
        pred_text for i, pred_text in enumerate(predictions_text_for_eval)
        if formatted_predictions_for_eval[i]['text_label'] is not None
    ]

    # Calculate accuracy
    accuracy = calculate_accuracy(valid_predictions_text, actual_labels_text)
    print(f"✅ 准确率 (仅含实际标签的样本): {accuracy:.4f}")

    # Collect bad cases
    bad_cases = [item for item in formatted_predictions_for_eval if item['text_label'] is not None and item['text_label'] != item['prediction']]
    print(f"❌ 发现 {len(bad_cases)} 个错误案例 (在有实际标签的样本中)。")

    # Prepare metrics.json content
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "num_samples_evaluated": len(actual_labels_text), # Number of samples with actual labels
        "num_correct": int(accuracy * len(actual_labels_text)) if actual_labels_text else 0,
        "num_errors": len(bad_cases),
    }

    # Save metrics.json
    metrics_file = os.path.join(output_dir, "metrics.json")
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"📊 评估指标已保存到: {metrics_file}")
    except Exception as e:
        print(f"❌ 保存 metrics.json 时发生错误: {e}")

    # Save bad_cases.jsonl
    if bad_cases:
        bad_cases_file = os.path.join(output_dir, "bad_cases.jsonl")
        try:
            with open(bad_cases_file, 'w', encoding='utf-8') as f:
                for entry in bad_cases:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"📝 错误案例已保存到: {bad_cases_file}")
        except Exception as e:
            print(f"❌ 保存 bad_cases.jsonl 时发生错误: {e}")
    else:
        print("✅ 没有错误案例需要保存。")

    print("📊 评估完成。")


def main():
    """
    主函数，处理命令行参数，执行 ONNX 模型预测或评估。
    """
    parser = argparse.ArgumentParser(
        description="使用 ONNX 模型对文本进行分类预测，并可选地进行评估。"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入数据集文件路径 (CSV, TSV, JSON 或 JSONL 格式)。"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="results_onnx",
        help="保存预测结果和评估指标的目录。"
    )
    parser.add_argument(
        "-m", "--model_dir",
        type=str,
        required=True,
        help="包含 ONNX 模型文件和 tokenizer 文件的目录。"
    )
    parser.add_argument(
        "--onnx_file",
        type=str,
        required=True, # ONNX file name is crucial for ORTModel
        help="ONNX 模型文件名 (e.g., 'model.onnx', 'model_quantized.onnx')。"
    )
    parser.add_argument(
        "-l", "--label_file",
        type=str,
        help="标签映射文件 (JSON格式)，包含数字ID到文本标签的映射。在评估模式下必需。"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="输入文件中包含文本数据的列名。"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="text_label",
        help="输入文件中包含实际文本标签的列名。"
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="开启评估模式。需要输入文件包含有效的标签列，并且--label_file必须指定。"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="用于 ONNX 推理的 GPU 设备 ID (0 表示第一个 GPU, -1 表示 CPU)。"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Transformer tokenizer 和 ONNX 模型处理的最大输入文本长度。"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批处理推理时使用的批次大小。"
    )

    args = parser.parse_args()

    # --- Parameter Validation ---
    if args.do_eval and not args.label_file:
        print("❌ 错误: 开启评估模式 (--do_eval) 时，必须指定标签映射文件 (--label_file)。")
        sys.exit(1)

    # --- Print Script Configuration ---
    print("=" * 60)
    print("✨ 开始执行 ONNX 文本分类任务 ✨")
    print("=" * 60)
    print(f"📊 输入文件: {args.input}")
    print(f"🎯 ONNX 模型目录: {args.model_dir}")
    print(f"📄 ONNX 文件名: {args.onnx_file}")
    print(f"📂 输出目录: {args.output_dir}")
    if args.label_file:
        print(f"🏷️ 标签文件: {args.label_file}")
    else:
        print("🏷️ 标签文件: 未指定 (无法将预测 ID 转换为文本，评估模式会受影响)")
    print(f"✏️ 文本列: '{args.text_column}'")
    print(f"🏷️ 实际标签列: '{args.label_column}'")
    print(f"🚀 评估模式: {'开启' if args.do_eval else '关闭'}")
    print(f"🖥️ 设备: {'GPU:' + str(args.device) if args.device >= 0 else 'CPU'}")
    print(f"📏 最大长度: {args.max_length}")
    print(f"🔢 批处理大小: {args.batch_size}")
    print("-" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    # actual_labels_text_ls contains the text labels from the user's specified label_column
    text_ls, actual_labels_text_ls, data_length, label_column_present = load_data(
        args.input, args.text_column, args.label_column
    )

    # If evaluation mode is enabled, but no label column is found, exit with an error.
    if args.do_eval and not label_column_present:
        print(f"❌ 错误: 开启了评估模式 (--do_eval)，但输入文件中未找到指定的标签列 '{args.label_column}'。")
        sys.exit(1)

    # Load label mapping (if label file is provided)
    id_to_label_map = None
    if args.label_file:
        id_to_label_map = load_label_map(args.label_file)
        # If label file is provided, but input has no label column, evaluation will be impacted.
        if not label_column_present:
            print("⚠️ 警告: 提供了标签文件，但输入文件没有找到指定的标签列。预测结果将尝试转换，但评估准确率无法进行。")

    # Load ONNX model and tokenizer
    tokenizer, model_runner = load_model_and_classifier(
        args.model_dir, args.onnx_file, args.device, args.max_length, args.batch_size
    )

    # Execute prediction
    # outputs is the raw output from the ONNX model (logits numpy array)
    outputs, start_time, end_time = batch_predict(model_runner, text_ls, args.batch_size)

    # Calculate QPS and execution time
    data_size = len(text_ls)
    execution_time = end_time - start_time
    qps = data_size / execution_time if execution_time > 0 else float('inf')
    print(f"\n--- ONNX 推理性能 ---")
    print(f"处理数据量: {data_size} 条")
    print(f"总执行时间: {execution_time:.4f} 秒")
    print(f"QPS: {qps:.2f}")
    print(f"-----------------------")

    # Format and save predictions
    # formatted_predictions_for_eval: contains {'text', 'text_label', 'prediction', 'score'}
    # predictions_text_for_eval: only contains 'prediction' list, used for direct accuracy calculation
    formatted_predictions_for_eval, predictions_text_for_eval = format_and_save_predictions(
        outputs, text_ls, actual_labels_text_ls, id_to_label_map, args.output_dir
    )

    # If evaluation mode is enabled, perform evaluation and save metrics
    if args.do_eval:
        evaluate_and_save_metrics(formatted_predictions_for_eval, predictions_text_for_eval, args.output_dir)

    print("\n🎉 任务执行完毕！ 🎉")


if __name__ == "__main__":
    main()
