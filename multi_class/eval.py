"""
测试脚本，用于评估分类模型的性能。
python src/eval.py ../train_data/test.csv ckpt/roberta_intention_cls_1w_tst/
python src/eval.py data/patient_intention_data_v2.1/test.json ckpt/patient_intention_cls2_aug  results/patientv2_aug7k
python src/eval.py data/patient_v2_3/test.jsonl ckpt/patient_v2_3  results/patientv23
python src/eval.py data/patient_v2_3/test.jsonl ckpt/patient_v2_3_aug  results/patientv23_aug
python src/eval.py data/patient_v2_3/test.jsonl ckpt/medbert_patient_v2_3/checkpoint-1500/  results/patientv23_med

"""
import os
import sys
import time
import json
from datetime import datetime

from datasets import Dataset
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def load_data(test_file):
    """
    加载测试数据。

    参数:
        test_file (str): 测试文件路径 (CSV 格式，包含 text 和 label 列)。

    返回:
        tuple: 包含文本列表和标签列表的元组。
    """
    if test_file.endswith('.csv') or test_file.endswith('.tsv'):
        df = pd.read_csv(test_file)
        text_ls = df["text"].tolist()
        label_ls = df["label"].tolist()
        text_label_ls = df["text_label"].tolist()
        
    elif test_file.endswith('.json') or test_file.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=test_file, split="train")
        text_ls = dataset["text"]
        label_ls = dataset["label"]
        text_label_ls = dataset["text_label"]
        
    else:
        raise ValueError("不支持的文件格式。请提供 CSV、TSV、JSON 或 JSONL 文件。")
    # 构建label map
    label2text = {}
    for label, label_text in zip(label_ls, text_label_ls):
        if label not in label2text:
            label2text[int(label)] = label_text
            
    return text_ls, label_ls,label2text


def load_model(model_id, use_ov=False):
    """
    加载意图分类模型。

    参数:
        model_id (str): 模型 ID 或路径。
        use_ov (bool): 是否使用 OpenVINO。默认为 False。

    返回:
        tuple: 包含 tokenizer 和模型对象的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    return tokenizer, hf_model


def create_classifier(model, tokenizer, device=0):
    """
    创建文本分类器 pipeline。

    参数:
        model: 模型对象。
        tokenizer: tokenizer 对象。
        device (int): 设备 ID。默认为 0。

    返回:
        pipeline: 文本分类器 pipeline。
    """
    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=device,truncation=True, max_length=256
    )
    return classifier


def predict(classifier, text_ls):
    """
    使用分类器进行预测。

    参数:
        classifier: 文本分类器 pipeline。
        text_ls (list): 文本列表。

    返回:
        list: 预测结果列表。
    """
    start = time.time()
    outputs = classifier(text_ls)
    end = time.time()
    # print("outputs",outputs)
    return outputs, start, end


def calculate_accuracy(predictions, actual_labels):
    """
    计算准确率。

    参数:
        predictions (list): 预测结果列表。
        actual_labels (list): 实际标签列表。

    返回:
        float: 准确率。
    """
    # print("predictions",predictions)
    # print("actual_labels", actual_labels)
    
    correct_predictions = sum(
       int(p)==int(a) for p, a in zip(predictions, actual_labels)
    )
    total_predictions = len(actual_labels)
    accuracy = correct_predictions / total_predictions
    return accuracy


def main():
    """
    主函数，用于加载模型、进行预测和评估性能。
    """
    test_file = sys.argv[1]
    model_id = sys.argv[2]
    outdir = sys.argv[3]
    os.makedirs(outdir, exist_ok = True)
    
    # 加载数据
    text_ls, label_ls,label2text = load_data(test_file)

    # 加载模型
    tokenizer, hf_model = load_model(model_id)

    # 创建分类器
    classifier = create_classifier(hf_model, tokenizer)

    # 进行预测
    outputs, start, end = predict(classifier, text_ls)
    predictions = [output["label"] for output in outputs]

    # 计算性能指标
    data_size = len(text_ls)
    dual = end - start
    qps = data_size / dual
    print("执行时间:", dual, "秒")
    print("QPS:", qps)

    # 计算准确率
    accuracy = calculate_accuracy(predictions, label_ls)
    print("准确率:", accuracy)

    # 保存评估指标到 metrics.json
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "qps": float(qps),
        "execution_time": float(dual),
        "data_size": data_size,
        "model_id": model_id,
        "test_file": test_file
    }
    metrics_file = os.path.join(outdir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存数据和bad_case
    data_ls = []
    bad_cases = []
    for text,pred, label in zip(text_ls, predictions, label_ls):
        # TODO: 处理标签 int改成实际的
        label = label2text[int(label)]
        pred = label2text[int(pred)]
        data = {'text': text, "label": label, "prediction": pred}
        data_ls.append(data)
        if label != pred:
            bad_cases.append(data)

    pred_file = os.path.join(outdir, "predictions.jsonl")
    bad_cases_file = os.path.join(outdir, "bad_cases.jsonl")
    
    ds = Dataset.from_list(data_ls)
    ds_bad = Dataset.from_list(bad_cases)
    ds.to_json(pred_file, force_ascii=False, lines=True)
    ds_bad.to_json(bad_cases_file, force_ascii=False, lines=True)
    


if __name__ == "__main__":
    main()
