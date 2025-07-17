'''
@author: jiaohuix
@file: inference.py
@date: 2025/07/17 14:39

# 仅预测 (不需要 label_file, 也不需要输入文件有 label 列)
python inference.py \
    --input data/test.jsonl \
    --output_dir results/prediction_only \
    --model_dir path/to/your/model \
    --text_column "text"

# 预测并尝试转换为文本标签 (需要 label_file)
python inference.py \
    --input data/test.jsonl \
    --output_dir results/prediction_with_mapping \
    --model_dir path/to/your/model \
    --label_file path/to/labels.json \
    --text_column "text"

# 评估模式 (需要 label_file 和输入文件包含 label_column)
python inference.py \
    --input data/test_with_labels.jsonl \
    --output_dir results/evaluation_run \
    --model_dir path/to/your/model \
    --label_file path/to/labels.json \
    --text_column "text" \
    --label_column "text_label" \
    --do_eval \
    --batch_size 32

python inference.py -i data/test.jsonl -o results -m ckpt -l ckpt/labels.json --do_eval --batch_size 32
'''
import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


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
        id_to_label_map = {int(v): k for k, v in label_data.items()}
        print(f"🏷️ 成功加载标签映射文件: {label_file}")
        print(f"   标签映射 (数字 -> 文本): {id_to_label_map}")
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
        if input_file.endswith('.csv') or input_file.endswith('.tsv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json') or input_file.endswith('.jsonl'):
            # 使用 read_json with lines=True for JSONL, and default for JSON
            try:
                dataset = load_dataset("json", data_files=input_file, split="train")
                df = dataset.to_pandas()
            except Exception as e:
                 print(f"尝试使用 load_dataset 加载 JSON/JSONL 失败: {e}")
                 print("尝试直接使用 pandas.read_json 加载...")
                 # 对于 JSON Lines, json.loads 可能更通用
                 if input_file.endswith('.jsonl'):
                     data = []
                     with open(input_file, 'r', encoding='utf-8') as f:
                         for line in f:
                             data.append(json.loads(line))
                     df = pd.DataFrame(data)
                 else: # 尝试作为普通 JSON 加载
                     df = pd.read_json(input_file)

        else:
            raise ValueError("不支持的文件格式。请提供 CSV, TSV, JSON 或 JSONL 文件。")

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


def load_model_and_classifier(model_dir, device=0, max_length=256, batch_size=16):
    """
    加载模型、tokenizer 并创建 pipeline。
    假设模型输出的是数字标签 ID。

    参数:
        model_dir (str): 模型目录或 Hugging Face 模型 ID。
        device (int): 使用的 GPU 设备 ID。
        max_length (int): Transformer 模型处理的最大输入长度。
        batch_size (int): 批处理大小。

    返回:
        pipeline: 配置好的文本分类 pipeline。
    """
    print(f"🧠 正在加载模型: {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # pipeline 会根据模型的 config 来确定是输出 ID 还是直接输出 label 文本
        # 如果模型配置有 id2label 映射，pipeline 会尝试使用它。
        # 如果没有，pipeline 通常会输出原始的 ID。
        # 我们在这里假设模型输出数字 ID
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=max_length,
            batch_size=batch_size,
            return_all_scores=False # 只返回最高分的那个预测
        )
        print(f"✅ 模型加载成功。设备: {'cuda:' + str(device) if device >= 0 else 'cpu'}, 最大长度: {max_length}, 批处理大小: {batch_size}")
        return classifier
    except Exception as e:
        print(f"❌ 加载模型或创建 pipeline 时发生错误: {e}")
        sys.exit(1)


def predict(classifier, text_ls):
    """
    使用分类器进行预测。

    参数:
        classifier: 文本分类器 pipeline。
        text_ls (list): 文本列表。

    返回:
        tuple: 包含预测结果列表、开始时间戳和结束时间戳的元组。
               预测结果的 'label' 字段应为数字 ID。
    """
    print("🚀 开始进行文本预测...")
    start_time = time.time()
    try:
        # pipeline 直接接受列表并处理批处理
        # 假设 pipeline 的输出是 [{'label': '0', 'score': 0.9}, ...]
        outputs = classifier(text_ls)
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        sys.exit(1)
    end_time = time.time()
    print("✅ 预测完成。")
    return outputs, start_time, end_time


def convert_prediction_to_text(outputs, id_to_label_map):
    """
    将模型的数字 ID 预测转换为文本标签，使用提供的映射。

    参数:
        outputs (list): 原始预测输出 [{'label': '0', 'score': 0.9}, ...]
        id_to_label_map (dict): 数字标签 ID 到文本标签的映射。

    返回:
        list: 包含转换后的文本标签和分数的列表。
              例如: [{'prediction_text': '类别文本', 'score': 0.9}, ...]
    """
    converted_predictions = []
    if not id_to_label_map:
        print("⚠️ 未提供标签映射，无法将预测 ID 转换为文本。将保留数字 ID。")
        # 如果没有映射，直接返回原始输出，但将 label 改为字符串方便后续处理
        for item in outputs:
            converted_predictions.append({
                'prediction_text': str(item['label']),
                'score': float(item['score'])
            })
        return converted_predictions

    for item in outputs:
        try:
            # 尝试将预测的 label 字符串转换为数字 ID
            pred_id = int(float(item['label']))
            # 使用映射查找文本标签
            pred_text = id_to_label_map.get(pred_id)

            if pred_text is None:
                print(f"⚠️ 警告: 预测的类 ID '{pred_id}' 在标签映射中未找到。将使用占位符 '未知类别'。")
                pred_text = "未知类别"

            converted_predictions.append({
                'prediction_text': pred_text,
                'score': float(item['score'])
            })
        except (ValueError, TypeError):
            # 如果预测的 label 不是有效的数字 ID
            print(f"⚠️ 警告: 预测输出的 label '{item['label']}' 无法解析为数字 ID。将使用占位符 '解析错误'。")
            converted_predictions.append({
                'prediction_text': "解析错误",
                'score': float(item['score'])
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
        outputs (list): 分类器返回的原始预测结果 (包含数字 label ID)。
        text_ls (list): 输入的文本列表。
        actual_labels_text_ls (list): 实际的文本标签列表。
        id_to_label_map (dict): 数字标签 ID 到文本标签的映射。
        output_dir (str): 保存结果的目录。

    返回:
        list: 格式化后的预测结果列表，用于评估。
              每个元素是一个字典: {'text': ..., 'text_label': ..., 'prediction': ..., 'score': ...}
    """
    print("💾 正在格式化预测结果并准备保存...")
    
    # 将预测的数字 ID 转换为文本标签
    converted_predictions_data = convert_prediction_to_text(outputs, id_to_label_map)

    formatted_predictions_for_eval = []
    predictions_text_for_eval = [] # 仅用于计算准确率的预测文本列表

    for i, text in enumerate(text_ls):
        pred_data = converted_predictions_data[i]
        predicted_label_text = pred_data['prediction_text']
        score = pred_data['score']

        # 获取实际标签文本
        actual_label_text = None
        if actual_labels_text_ls and i < len(actual_labels_text_ls):
            actual_label_text = actual_labels_text_ls[i]

        formatted_predictions_for_eval.append({
            "text": text,
            "text_label": actual_label_text, # 保存原始实际标签文本
            "prediction": predicted_label_text, # 保存转换后的预测文本
            "score": score
        })
        predictions_text_for_eval.append(predicted_label_text)

    # 保存 prediction.jsonl
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

    actual_labels_text = [item['text_label'] for item in formatted_predictions_for_eval]

    # 计算准确率
    accuracy = calculate_accuracy(predictions_text_for_eval, actual_labels_text)
    print(f"✅ 准确率: {accuracy:.4f}")

    # 收集 bad cases
    bad_cases = [item for item in formatted_predictions_for_eval if item['text_label'] != item['prediction']]
    print(f"❌ 发现 {len(bad_cases)} 个错误案例。")

    # 准备 metrics.json 内容
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "num_samples": len(formatted_predictions_for_eval),
        "num_correct": int(accuracy * len(formatted_predictions_for_eval)),
        "num_errors": len(bad_cases),
    }

    # 保存 metrics.json
    metrics_file = os.path.join(output_dir, "metrics.json")
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"📊 评估指标已保存到: {metrics_file}")
    except Exception as e:
        print(f"❌ 保存 metrics.json 时发生错误: {e}")

    # 保存 bad_cases.jsonl
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
    主函数，处理命令行参数，执行预测或评估。
    """
    parser = argparse.ArgumentParser(
        description="对文本进行分类预测，并可选地进行评估。"
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
        default="results",
        help="保存预测结果和评估指标的目录。"
    )
    parser.add_argument(
        "-m", "--model_dir",
        type=str,
        required=True,
        help="模型目录或 Hugging Face 模型 ID。"
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
        help="用于推理的 GPU 设备 ID (0 表示第一个 GPU, -1 表示 CPU)。"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Transformer 模型处理的最大输入文本长度。"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批处理推理时使用的批次大小。"
    )

    args = parser.parse_args()

    # --- 参数校验 ---
    if args.do_eval and not args.label_file:
        print("❌ 错误: 开启评估模式 (--do_eval) 时，必须指定标签映射文件 (--label_file)。")
        sys.exit(1)

    # --- 打印脚本配置信息 ---
    print("=" * 60)
    print("✨ 开始执行文本分类任务 ✨")
    print("=" * 60)
    print(f"📊 输入文件: {args.input}")
    print(f"🎯 模型路径: {args.model_dir}")
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

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    # actual_labels_text_ls 包含的是用户指定的 label_column 中的文本标签
    text_ls, actual_labels_text_ls, data_length, label_column_present = load_data(
        args.input, args.text_column, args.label_column
    )

    # 如果开启评估模式，但没有标签列，则发出警告并可能退出或继续无评估
    if args.do_eval and not label_column_present:
        print(f"❌ 错误: 开启了评估模式 (--do_eval)，但输入文件中未找到指定的标签列 '{args.label_column}'。")
        sys.exit(1)

    # 加载标签映射（如果提供了标签文件）
    id_to_label_map = None
    if args.label_file:
        id_to_label_map = load_label_map(args.label_file)
        # 如果提供了标签文件，但输入文件没有对应的标签列，那么评估会失效，但预测仍然可以尝试转换
        if not label_column_present:
             print("⚠️ 警告: 提供了标签文件，但输入文件没有找到指定的标签列。预测结果将尝试转换，但评估准确率无法进行。")

    # 加载模型和分类器
    classifier = load_model_and_classifier(args.model_dir, args.device, args.max_length, args.batch_size)

    # 执行预测
    # outputs 是原始输出，其中 'label' 是模型预测的数字 ID 字符串
    outputs, start_time, end_time = predict(classifier, text_ls)

    # 计算 QPS 和执行时间
    data_size = len(text_ls)
    execution_time = end_time - start_time
    qps = data_size / execution_time if execution_time > 0 else float('inf')
    print(f"\n--- 推理性能 ---")
    print(f"处理数据量: {data_size} 条")
    print(f"总执行时间: {execution_time:.4f} 秒")
    print(f"QPS: {qps:.2f}")
    print(f"------------------")

    # 格式化并保存预测结果
    # formatted_predictions_for_eval: 包含 {'text', 'text_label', 'prediction', 'score'}
    # predictions_text_for_eval: 仅包含 'prediction' 列表，用于直接计算准确率
    formatted_predictions_for_eval, predictions_text_for_eval = format_and_save_predictions(
        outputs, text_ls, actual_labels_text_ls, id_to_label_map, args.output_dir
    )

    # 如果开启评估模式，则进行评估并保存指标
    if args.do_eval:
        evaluate_and_save_metrics(formatted_predictions_for_eval, predictions_text_for_eval, args.output_dir)

    print("\n🎉 任务执行完毕！ 🎉")


if __name__ == "__main__":
    main()
