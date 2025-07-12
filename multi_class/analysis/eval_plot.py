"""
批量测试脚本，调用 FastAPI 接口进行文本分类，保存结果，并进行分析和可视化。
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from datasets import load_dataset
from weasyprint import HTML
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# 配置中文字体支持
try:
    # 尝试设置全局中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 检查是否有可用的中文字体
    chinese_fonts = [f for f in fm.findSystemFonts() if '\\chinese' in f.lower() 
                     or 'simhei' in f.lower() 
                     or 'yahei' in f.lower() 
                     or 'simsun' in f.lower()]
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'].insert(0, chinese_fonts[0])
        logging.info(f"使用中文字体: {chinese_fonts[0]}")
    else:
        logging.warning("未找到系统中文字体，可能会影响图表中文显示")
except Exception as e:
    logging.warning(f"设置中文字体时出错: {e}")

# 默认的 FastAPI 服务 URL
DEFAULT_SERVER_URL = "http://localhost:8000/predict/"


def call_api(server_url: str, texts: List[str]) -> Optional[List[Dict[str, Any]]]:
    """
    调用 FastAPI 接口，获取文本列表的预测结果。

    Args:
        server_url (str): FastAPI 服务的 URL。
        texts (List[str]): 要分类的文本列表。

    Returns:
        Optional[List[Dict[str, Any]]]: 预测结果列表，每个结果是一个字典，包含文本、标签、
        类别索引和分数。如果请求失败，则返回 None。
    """
    headers = {"Content-Type": "application/json"}
    data = {"texts": texts}

    try:
        response = requests.post(
            server_url, headers=headers, data=json.dumps(data)
        )
        response.raise_for_status()  # 为错误的响应（4xx 或 5xx）引发 HTTPError
        return response.json()["predictions"][0]  # 访问预测结果列表
    except requests.exceptions.RequestException as e:
        logging.error(f"API 请求失败: {e}")
        return None


def setup_logging(log_dir: str) -> None:
    """
    配置日志记录。

    Args:
        log_dir (str): 日志目录。
    """
    # 确保 logs 目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置日志
    log_file_path = os.path.join(
        log_dir, f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSON 数据文件。

    Args:
        file_path (str): JSON 数据文件路径。

    Returns:
        List[Dict[str, Any]]: 数据列表，每个元素是一个字典。
    """
    
    try:
        logging.info(f"正在读取数据文件: {file_path}")
        data = load_dataset("json", data_files=file_path, split="train")
        # with open(file_path, "r", encoding="utf-8") as f:
        #     data = json.load(f)

        # 将数据集转换为字典
        data_dict = data.to_dict()
        
        # 使用列表推导式将数据转换为 [{}, {}, ...] 的格式
        data_list = [{key: value[i] for key, value in data_dict.items()} for i in range(len(data_dict[list(data_dict.keys())[0]]))]

        logging.info(f"成功加载数据，共{len(data_list)}条记录")
        return data_list
    except Exception as e:
        logging.error(f"读取数据文件失败: {e}")
        return []


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    保存推理结果到 JSON 文件。

    Args:
        results (List[Dict[str, Any]]): 推理结果列表。
        output_file (str): 输出文件路径。
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info(f"正在保存结果到: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logging.error(f"保存结果失败: {e}")


def batch_test(
    input_file: str,
    server_url: str,
    batch_size: int = 32,
    delay: int = 1,
) -> List[Dict[str, Any]]:
    """
    批量测试函数。

    Args:
        input_file (str): 输入文件路径。
        server_url (str): FastAPI 服务的 URL。
        batch_size (int): 批处理大小。默认为 32。
        delay (int): 批次之间的延迟（秒）。默认为 1。

    Returns:
        List[Dict[str, Any]]: 包含推理结果的列表。
    """
    # 读取输入数据
    data = load_data(input_file)
    if not data:
        return []

    # 准备输出结果
    results = []

    # 记录开始时间
    start_time = time.time()
    logging.info(f"开始批量测试，共{len(data)}条数据")

    # 批量处理数据
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        texts = [item["text"] for item in batch]
        print("texts",texts, len(texts))

        logging.info(
            f"正在处理批次 {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}, 包含{len(texts)}条文本"
        )

        try:
            # 调用 API 进行推理
            api_response = call_api(server_url, texts)

            if api_response:
                # 将推理结果与原始数据结构合并
                for j, prediction in enumerate(api_response):
                    # 如果原始数据包含 intention 字段，保留它
                    original_item = batch[j]

                    pred = prediction["label"]
                    # if pred == "其他":
                    #     pred = "咨询"  # 避免硬编码，可以考虑在配置中定义映射关系

                    result_item = {
                        "text": prediction["text"],
                        "intention": pred,  # 使用 API 返回的 label 作为 intention
                    }

                    # 可选：添加额外的 API 返回信息供分析
                    result_item["score"] = prediction["score"]
                    result_item["class_idx"] = prediction["class_idx"]

                    # 可选：保留原始数据中的 intention 用于比较
                    if "text_label" in original_item:
                        result_item["original_intention"] = original_item[
                            "text_label"
                        ]

                    results.append(result_item)

                logging.info(f"批次 {i//batch_size + 1} 完成处理")
            else:
                logging.error(f"批次 {i//batch_size + 1} 调用 API 失败，跳过此批次")
        except Exception as e:
            logging.error(f"处理批次 {i//batch_size + 1} 时出错: {e}")

        # 在批次之间添加短暂延迟，避免 API 限制
        if i + batch_size < len(data):
            time.sleep(delay)

    # 记录总耗时
    elapsed_time = time.time() - start_time
    logging.info(f"批量测试完成，共处理{len(results)}条数据，耗时{elapsed_time:.2f}秒")

    return results


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """加载推理结果数据（支持json、jsonl）"""
    try:
        logging.info(f"正在读取结果文件: {file_path}")
        ds = load_dataset('json', data_files=file_path, split="train")
        # 将 Dataset 转换为列表
        data_ls = ds.to_list()
        logging.info(f"成功加载数据，共{len(data_ls)}条记录")
        return data_ls
    except Exception as e:
        logging.error(f"读取结果文件失败: {e}")
        return []




def prepare_data(data, redirect_intentions=False):
    """准备数据用于分析
    
    参数:
        data: 输入数据列表
        redirect_intentions: 是否进行类别重定向，合并特定类别类别
    """
    # 提取预测和真实标签
    y_true = []
    y_pred = []
    texts = []
    scores = []
    
    # 如果启用类别重定向，定义映射关系
    redirect_mapping = {}

    
    # 检查数据中是否包含original_intention字段
    if 'original_intention' in data[0]:
        for item in data:
            # 获取原始类别和预测类别
            original_intention = item.get('original_intention', '')
            predicted_intention = item.get('intention', '')
          
            y_true.append(original_intention)
            y_pred.append(predicted_intention)
            texts.append(item.get('text', ''))
            scores.append(item.get('score', 0))
    else:
        logging.warning("数据中没有original_intention字段，无法进行性能评估")
        return None, None, None, None
    
    return y_true, y_pred, texts, scores


def calculate_metrics(y_true, y_pred):
    """计算性能评估指标"""
    # 获取所有唯一的类别标签
    all_labels = sorted(list(set(y_true + y_pred)))
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"准确率 (Accuracy): {accuracy:.4f}")
    
    # 计算宏平均精确率、召回率、F1值
    precision_macro = precision_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0)
    
    logging.info(f"宏平均精确率 (Precision): {precision_macro:.4f}")
    logging.info(f"宏平均召回率 (Recall): {recall_macro:.4f}")
    logging.info(f"宏平均F1值 (F1): {f1_macro:.4f}")
    
    # 计算每个类别的精确率、召回率、F1值
    report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True)
    
    # 返回结果
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'report': report,
        'labels': all_labels
    }
    
    return metrics

def create_confusion_matrix(y_true, y_pred, labels, output_dir='.'):
    """创建并保存混淆矩阵"""
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 创建DataFrame用于可视化
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # 绘制热力图
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵', fontsize=16)
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 保存图像
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    logging.info(f"混淆矩阵已保存到: {confusion_matrix_path}")
    plt.close()

def create_class_distribution(y_true, y_pred, labels, output_dir='.'):
    """创建类别分布图"""
    plt.figure(figsize=(12, 8))
    
    # 计算真实和预测类别的分布
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    # 准备数据
    df = pd.DataFrame({
        '真实分布': [true_counts.get(label, 0) for label in labels],
        '预测分布': [pred_counts.get(label, 0) for label in labels]
    }, index=labels)
    
    # 绘制柱状图
    ax = df.plot(kind='bar', figsize=(14, 8))
    plt.title('类别分布对比', fontsize=16)
    plt.ylabel('样本数量', fontsize=14)
    plt.xlabel('类别', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    
    # 保存图像
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plt.tight_layout()
    plt.savefig(class_dist_path)
    logging.info(f"类别分布图已保存到: {class_dist_path}")
    plt.close()

def create_score_distribution(scores, y_true, y_pred, output_dir='.'):
    """创建置信度分数分布图，区分正确和错误预测"""
    plt.figure(figsize=(10, 6))
    
    # 确定预测是否正确
    correct = [score for score, true, pred in zip(scores, y_true, y_pred) if true == pred]
    incorrect = [score for score, true, pred in zip(scores, y_true, y_pred) if true != pred]
    
    # 设置直方图参数
    bins = np.linspace(0, 1, 20)
    
    # 绘制直方图
    plt.hist(correct, bins=bins, alpha=0.5, label='正确预测', color='green')
    plt.hist(incorrect, bins=bins, alpha=0.5, label='错误预测', color='red')
    
    plt.title('预测置信度分布', fontsize=16)
    plt.xlabel('置信度分数', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    score_dist_path = os.path.join(output_dir, 'score_distribution.png')
    plt.tight_layout()
    plt.savefig(score_dist_path)
    logging.info(f"置信度分布图已保存到: {score_dist_path}")
    plt.close()

def create_class_performance(metrics, output_dir='.'):
    """创建每个类别的性能图表"""
    plt.figure(figsize=(14, 10))
    
    report = metrics['report']
    # 过滤掉非类别的键
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # 收集每个类别的精确率、召回率和F1值
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1 = [report[cls]['f1-score'] for cls in classes]
    support = [report[cls]['support'] for cls in classes]
    
    # 创建DataFrame
    df = pd.DataFrame({
        '精确率': precision,
        '召回率': recall,
        'F1值': f1,
        '支持度': support
    }, index=classes)
    
    # 排序，按样本数量降序
    df = df.sort_values('支持度', ascending=False)
    
    # 绘制条形图
    ax = df[['精确率', '召回率', 'F1值']].plot(kind='bar', figsize=(14, 10))
    plt.title('各类别性能指标', fontsize=16)
    plt.ylabel('分数', fontsize=14)
    plt.xlabel('类别', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylim(0, 1.1)  # 设置y轴范围
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7)
    
    # 在柱状图上方显示支持度
    for i, support_val in enumerate(df['支持度']):
        plt.text(i, 1.05, f'n={support_val}', ha='center', fontsize=8)
    
    # 保存图像
    class_perf_path = os.path.join(output_dir, 'class_performance.png')
    plt.tight_layout()
    plt.savefig(class_perf_path)
    logging.info(f"类别性能图已保存到: {class_perf_path}")
    plt.close()

def analyze_errors(y_true, y_pred, texts, scores, output_dir='.'):
    """分析错误预测并保存到CSV文件"""
    errors = []
    
    for true, pred, text, score in zip(y_true, y_pred, texts, scores):
        if true != pred:
            errors.append({
                '文本': text,
                '真实标签': true,
                '预测标签': pred,
                '置信度': score
            })
    
    if errors:
        # 保存错误分析
        errors_df = pd.DataFrame(errors)
        error_analysis_path = os.path.join(output_dir, 'error_analysis.csv')
        errors_df.to_csv(error_analysis_path, index=False, encoding='utf-8-sig')
        logging.info(f"错误分析已保存到: {error_analysis_path}，共{len(errors)}条错误")
        
        # 计算错误最多的类别对
        error_pairs = [f"{true} -> {pred}" for true, pred in zip(errors_df['真实标签'], errors_df['预测标签'])]
        pair_counts = Counter(error_pairs).most_common(10)
        
        logging.info("错误最多的10个类别对:")
        for pair, count in pair_counts:
            logging.info(f"  {pair}: {count}条")
            
        # 绘制错误最多的类别对
        plt.figure(figsize=(12, 6))
        pairs = [pair for pair, _ in pair_counts]
        counts = [count for _, count in pair_counts]
        
        plt.barh(pairs, counts, color='salmon')
        plt.title('错误预测最多的类别对', fontsize=16)
        plt.xlabel('错误数量', fontsize=14)
        plt.ylabel('真实标签 -> 预测标签', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # 在条形图上添加数值
        for i, count in enumerate(counts):
            plt.text(count + 0.5, i, str(count), va='center', fontsize=10)
        
        # 保存图像
        error_pairs_path = os.path.join(output_dir, 'error_pairs.png')
        plt.tight_layout()
        plt.savefig(error_pairs_path)
        logging.info(f"错误类别对图已保存到: {error_pairs_path}")
        plt.close()
    else:
        logging.info("没有发现预测错误！")


def generate_pdf_from_html(html_path, pdf_path):
    try:
        HTML(html_path).write_pdf(pdf_path)
        logging.info(f"PDF报告已生成: {pdf_path}")
        return True
    except Exception as e:
        logging.error(f"使用WeasyPrint生成PDF失败: {e}")
        return False
    
def generate_html_report(metrics, output_dir='.'):
    """生成HTML格式的完整报告"""
    report_path = os.path.join(output_dir, 'model_evaluation_report.html')
    pdf_path = os.path.join(output_dir, 'model_evaluation_report.pdf')
    
    report = metrics['report']
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # 创建类别性能表格HTML
    class_rows = []
    for cls in classes:
        class_rows.append(f"""
        <tr>
            <td>{cls}</td>
            <td>{report[cls]['precision']:.4f}</td>
            <td>{report[cls]['recall']:.4f}</td>
            <td>{report[cls]['f1-score']:.4f}</td>
            <td>{int(report[cls]['support'])}</td>
        </tr>""")
    
    class_table = ''.join(class_rows)
    
    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>模型评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .metric-name {{ font-size: 14px; color: #666; }}
            .image-container {{ margin: 20px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>模型评估报告</h1>
        <p>评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>总体性能指标</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <div class="metric-value">{metrics['accuracy']:.4f}</div>
                <div class="metric-name">准确率 (Accuracy)</div>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <div class="metric-value">{metrics['precision_macro']:.4f}</div>
                <div class="metric-name">宏平均精确率 (Precision)</div>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <div class="metric-value">{metrics['recall_macro']:.4f}</div>
                <div class="metric-name">宏平均召回率 (Recall)</div>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px;">
                <div class="metric-value">{metrics['f1_macro']:.4f}</div>
                <div class="metric-name">宏平均F1值</div>
            </div>
        </div>
        
        <h2>各类别性能指标</h2>
        <table>
            <tr>
                <th>类别</th>
                <th>精确率 (Precision)</th>
                <th>召回率 (Recall)</th>
                <th>F1值</th>
                <th>样本数量</th>
            </tr>
            {class_table}
        </table>
        
        <h2>可视化结果</h2>
        
        <h3>混淆矩阵</h3>
        <div class="image-container">
            <img src="confusion_matrix.png" alt="混淆矩阵">
        </div>
        
        <h3>类别分布</h3>
        <div class="image-container">
            <img src="class_distribution.png" alt="类别分布">
        </div>
        
        <h3>各类别性能</h3>
        <div class="image-container">
            <img src="class_performance.png" alt="各类别性能">
        </div>
        
        <h3>预测置信度分布</h3>
        <div class="image-container">
            <img src="score_distribution.png" alt="预测置信度分布">
        </div>
        
        <h3>错误预测分析</h3>
        <div class="image-container">
            <img src="error_pairs.png" alt="错误预测最多的类别对">
        </div>
        
        <p><strong>注意:</strong> 完整的错误分析详情请参见 error_analysis.csv 文件。</p>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    

    try:
        generate_pdf_from_html(report_path, pdf_path)
        # pdfkit.from_file(report_path, pdf_path, configuration=configuration)
        logging.info(f"PDF报告已生成: {pdf_path}")
    except Exception as e:
        logging.error(f"生成PDF报告失败: {e}")

    logging.info(f"HTML评估报告已生成: {report_path}")

def create_comparison_matrix(y_true, y_pred, metrics, output_dir='.'):
    """创建类别间相似性矩阵，找出容易混淆的类别"""
    report = metrics['report']
    labels = metrics['labels']
    
    # 创建空矩阵
    size = len(labels)
    similarity = np.zeros((size, size))
    
    # 计算每对类别间的相似性（混淆程度）
    # 这里用一个简单的启发式方法，如果类别i被错误分类为类别j的次数与被正确分类的次数的比率
    # 真实的混淆矩阵可以通过混淆矩阵获得
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 计算相对混淆程度（将每一行标准化，以便比较）
    norm_cm = np.zeros_like(cm, dtype=float)
    for i in range(size):
        row_sum = np.sum(cm[i])
        if row_sum > 0:
            norm_cm[i] = cm[i] / row_sum
    
    # 创建相似性矩阵的热力图
    plt.figure(figsize=(14, 12))
    similarity_df = pd.DataFrame(norm_cm, index=labels, columns=labels)
    
    # 绘制热力图
    mask = np.eye(size, dtype=bool)  # 创建对角线掩码
    sns.heatmap(similarity_df, cmap='YlGnBu', annot=True, fmt='.2f', mask=mask)
    plt.title('类别间混淆程度（非对角元素）', fontsize=16)
    plt.ylabel('真实类别', fontsize=14)
    plt.xlabel('预测类别', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 保存图像
    similarity_matrix_path = os.path.join(output_dir, 'similarity_matrix.png')
    plt.tight_layout()
    plt.savefig(similarity_matrix_path)
    logging.info(f"类别相似性矩阵已保存到: {similarity_matrix_path}")
    plt.close()

def analyze_results(result_file:str, output_dir:str, redirect_intentions:bool=False):
    # 设置输出目录
    print("output_dir:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置结果文件路径
    
    # 加载数据
    data = load_results(result_file)
    if not data:
        return
    
    # 准备数据
    y_true, y_pred, texts, scores = prepare_data(data, redirect_intentions=redirect_intentions)
    if y_true is None:
        return
    
    # 计算评估指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 生成混淆矩阵
    create_confusion_matrix(y_true, y_pred, metrics['labels'], output_dir)
    
    # 生成类别分布图
    create_class_distribution(y_true, y_pred, metrics['labels'], output_dir)
    
    # 生成置信度分布图
    create_score_distribution(scores, y_true, y_pred, output_dir)
    
    # 生成类别性能图
    create_class_performance(metrics, output_dir)
    
    # 分析错误
    analyze_errors(y_true, y_pred, texts, scores, output_dir)
    
    # 创建类别相似性矩阵
    create_comparison_matrix(y_true, y_pred, metrics, output_dir)
    
    # 生成HTML报告
    generate_html_report(metrics, output_dir)
    
    logging.info(f"分析完成，所有结果已保存到: {output_dir}")


def main():
    """
    主函数，用于解析命令行参数、配置日志、执行批量测试、保存结果并进行分析。
    """
    parser = argparse.ArgumentParser(
        description="批量测试脚本，用于调用 FastAPI 接口进行文本分类，并进行分析和可视化。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入 JSON 数据文件路径",
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="批处理大小"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help="FastAPI 服务 URL",
    )
    parser.add_argument(
        "--delay", type=int, default=1, help="批次之间的延迟（秒）"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="日志文件目录",
    )
    parser.add_argument(
        "--redirect",
        action="store_true",
        help="启用类别重定向，合并特定类别类别",
    )
    parser.add_argument(
        "--analysis_output_dir",
        type=str,
        default="analysis_results",
        help="分析结果输出目录",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="推理结果 JSON 文件路径 (如果已存在，则跳过推理)",
    )
    parser.add_argument(
        "--call_shell",
        type=str,
        default=None,
        help="调用shell命令",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.log_dir)
    output_file = os.path.join(args.analysis_output_dir, "results.json")
    logging.info("===== 批量推理测试开始 =====")
    logging.info(f"输入文件: {args.input_file}")
    logging.info(f"输出文件: {output_file}")
    logging.info(f"批处理大小: {args.batch_size}")
    logging.info(f"FastAPI 服务 URL: {args.server_url}")
    logging.info(f"批次之间延迟: {args.delay}秒")
    logging.info(f"日志目录: {args.log_dir}")
    logging.info(f"分析结果输出目录: {args.analysis_output_dir}")
    logging.info(f"是否启用类别重定向: {args.redirect}")

    # 1. 批量推理
    results = []
    if args.result_file and os.path.exists(args.result_file):
        logging.info(f"发现已存在的推理结果文件: {args.result_file}，跳过推理过程")
        results = load_results(args.result_file)
    else:
        logging.info("开始调用API进行批量推理...")
        results = batch_test(
            args.input_file, args.server_url, args.batch_size, args.delay
        )
        if results:
            save_results(results, output_file)
        else:
            logging.error("测试失败，未生成结果")

    # 2. 结果分析和可视化
    if results:
        logging.info("开始分析推理结果...")
        analyze_results(output_file, args.analysis_output_dir, args.redirect)
    else:
        logging.error("没有可分析的结果，请检查推理过程是否成功")

    # 3. 调用shell命令
    if args.call_shell:
        logging.info(f"开始调用shell命令: {args.call_shell}")
        try:
            # 使用 subprocess.run 执行 shell 命令
            result = subprocess.run(args.call_shell, shell=True, capture_output=True, text=True, check=True)

            # 打印命令的输出
            logging.info(f"Shell 命令输出:\n{result.stdout}")

            # 如果命令有错误输出，也打印出来
            if result.stderr:
                logging.error(f"Shell 命令错误:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            # 如果命令返回非零退出代码，则捕获异常
            logging.error(f"Shell 命令执行失败，返回码: {e.returncode}")
            logging.error(f"错误信息:\n{e.stderr}")
        except FileNotFoundError:
            logging.error(f"Shell 命令未找到: {args.call_shell}")
        except Exception as e:
            logging.error(f"调用shell命令时发生错误: {e}")

    logging.info("===== 批量推理测试结束 =====")


if __name__ == "__main__":
    # 为了避免循环依赖，将 seaborn 的导入放在 main 函数中
    import seaborn as sns
    main()