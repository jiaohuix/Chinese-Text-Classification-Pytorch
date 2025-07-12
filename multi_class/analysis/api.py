"""
FastAPI 服务，用于提供类别分类预测接口。
"""

import time
from typing import List, Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from optimum.onnxruntime import ORTModelForSequenceClassification
import argparse

# 创建 FastAPI 应用
app = FastAPI()

# 模型配置
# USE_OV = False  # 是否使用 OpenVINO
LABELS = [  # 类别标签列表
    "其他",
]

# 全局变量，用于存储模型和 tokenizer
tokenizer = None
hf_model = None
classifier = None


# 加载模型和 tokenizer
def load_model(model_id: str, use_ov: bool):
    """
    加载类别分类模型和 tokenizer。

    参数:
        model_id (str): 模型 ID 或路径。
        use_ov (bool): 是否使用 OpenVINO。

    返回:
        tuple: 包含 tokenizer 和模型对象的元组。
    """
    global tokenizer, hf_model, classifier  # 声明使用全局变量

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # if use_ov:
    #     print("使用 OpenVINO...")
    #     hf_model = ORTModelForSequenceClassification.from_pretrained(
    #         model_id, file_name="model.onnx"
    #     )
    # else:
    #     hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)

    hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # 创建文本分类器 pipeline
    classifier = pipeline(
        "text-classification", model=hf_model, tokenizer=tokenizer, device=0,
        max_length=256,  # 设置最大序列长度
        truncation=True  # 启用截断以确保不超过max_length
    )  
    return tokenizer, hf_model, classifier


# 定义请求和响应模型
class PredictionRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    predictions: List[List[Dict[str, Union[str, float, int]]]]


# 定义预测接口
@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    预测文本列表的类别。

    参数:
        request (PredictionRequest): 包含文本列表的请求对象。

    返回:
        PredictionResponse: 包含预测结果的响应对象。
    """
    texts = request.texts

    start_time = time.time()
    try:
        outputs = classifier(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
    end_time = time.time()

    predictions = []
    for i, output in enumerate(outputs):
        class_idx = int(output["label"].split("_")[-1])  # 提取类别索引
        label = LABELS[class_idx]
        predictions.append(
            {
                "text": texts[i],
                "label": label,
                "class_idx": class_idx,
                "score": output["score"],
            }
        )

    execution_time = end_time - start_time
    qps = len(texts) / execution_time
    print(f"推理时间: {execution_time:.4f} 秒")
    print(f"QPS: {qps:.2f}")

    return PredictionResponse(predictions=[predictions])


def main():
    """
    主函数，用于解析命令行参数并启动 FastAPI 服务。
    """
    parser = argparse.ArgumentParser(
        description="FastAPI 服务，用于提供类别分类预测接口。"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,  # 必须指定 model_id
        help="模型 ID 或路径",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务监听的 host"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="服务监听的端口"
    )
    args = parser.parse_args()

    # 加载模型
    load_model(args.model_id, USE_OV)

    # 启动 FastAPI 服务
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
