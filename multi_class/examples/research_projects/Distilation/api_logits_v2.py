'''
提供预训练模型的蒸馏logits的服务。
输入：texts list[text]
输出：logits list[list[float]]
基于fastapi实现

v0.1.0 随机生成float16的logits
v0.2.0 用训练好的bert的logits提供服务 (当前版本)
v0.3.0 用llm提供logits蒸馏服务
v1.0.0 正式API服务
'''

import os
import time
import torch
import numpy as np
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import logging
from tqdm import tqdm
import torch.nn.functional as F

# FastAPI相关
from fastapi import FastAPI, HTTPException
import uvicorn

# Transformers相关
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义请求和响应的数据模型
class TextRequest(BaseModel):
    texts: List[str]
    with_all_scores: Optional[bool] = False  # 新增参数，默认为False

class LogitsResponse(BaseModel):
    logits: List[List[float]]

class ScoreItem(BaseModel):
    """单个类别的得分项"""
    label: str
    score: float = Field(..., ge=0.0, le=1.0)

class PredictItem(BaseModel):
    """单个预测结果"""
    text: str
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    all_scores: Optional[List[ScoreItem]] = None  # 允许为None

class PredictResponse(BaseModel):
    """预测响应模型"""
    predictions: List[PredictItem]

# 模型缓存
class ModelCache:
    def __init__(self, model_path: str, device: str = None, batch_size: int = 16):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.config = None
        self.num_classes = None
        self.result_cache = {}  # 简单的结果缓存
        self.max_cache_size = 10000  # 最大缓存条目数
        
        self.load_model()
        
    def load_model(self):
        """加载模型、分词器和配置"""
        logger.info(f"Loading model from {self.model_path}")
        start_time = time.time()
        
        try:
            self.config = AutoConfig.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                config=self.config
            )
            
            # 将模型移动到指定设备
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            # 获取类别数量
            self.num_classes = self.config.num_labels
            
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
            logger.info(f"Number of classes: {self.num_classes}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Batch size: {self.batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_logits(self, texts: List[str]) -> List[List[float]]:
        """获取文本的logits，支持批处理"""
        # 首先检查缓存
        cached_results = []
        texts_to_process = []
        
        for text in texts:
            if text in self.result_cache:
                cached_results.append((text, self.result_cache[text]))
            else:
                texts_to_process.append(text)
        
        # 如果所有文本都在缓存中，直接返回缓存的结果
        if not texts_to_process:
            logger.info(f"All {len(texts)} texts found in cache")
            # 按原始顺序返回结果
            return [self.result_cache[text] for text in texts]
        
        logger.info(f"Processing {len(texts_to_process)} texts in batches")
        all_logits = []
        
        # 分批处理
        for i in range(0, len(texts_to_process), self.batch_size):
            batch_texts = texts_to_process[i:i+self.batch_size]
            batch_logits = self._process_batch(batch_texts)
            all_logits.extend(batch_logits)
            
            # 更新缓存
            for text, logits in zip(batch_texts, batch_logits):
                self.result_cache[text] = logits
        
        # 管理缓存大小
        self._manage_cache_size()
        
        # 按原始顺序重建结果
        result = []
        processed_idx = 0
        
        for text in texts:
            if text in self.result_cache:
                result.append(self.result_cache[text])
            else:
                # 这种情况理论上不应该出现
                logger.warning(f"Text not found in cache after processing: {text[:30]}...")
                # 使用零向量作为回退
                result.append([0.0] * self.num_classes)
        
        return result
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """处理一批文本，返回logits"""
        with torch.no_grad():
            # 对文本进行编码
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=256,  # 可以根据需要调整
                return_tensors="pt"
            ).to(self.device)
            
            # 获取模型输出
            outputs = self.model(**inputs)
            
            # 提取logits
            logits = outputs.logits.cpu().numpy()
            
            # 转换为列表
            return logits.tolist()
    
    def _manage_cache_size(self):
        """管理缓存大小，如果超过限制则删除最早的条目"""
        if len(self.result_cache) > self.max_cache_size:
            # 删除20%的最早条目
            remove_count = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.result_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.result_cache[key]
            logger.info(f"Cache cleanup: removed {remove_count} oldest entries")

# 创建FastAPI应用
app = FastAPI(
    title="Knowledge Distillation API v2",
    description="提供预训练模型的logits蒸馏服务",
    version="2.0.0"
)

# 全局变量
model_cache = None
model_path = os.environ.get("MODEL_PATH", "ckpt/patient_intention/v25/v25_doubao_fgm/ckpt_v0620_fgm/")
device = os.environ.get("DEVICE", None)  # 自动选择
batch_size = int(os.environ.get("BATCH_SIZE", 16))

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_cache
    try:
        model_cache = ModelCache(model_path, device, batch_size)
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {str(e)}")
        # 应用将继续运行，但API调用会失败

@app.post("/v1/knowledge/logits", response_model=LogitsResponse)
async def distill_logits(request: TextRequest):
    """
    使用预训练模型为输入的文本列表生成logits
    :param request: 包含文本列表的请求
    :return: 每个文本对应的logits列表
    """
    global model_cache
    
    if model_cache is None:
        try:
            model_cache = ModelCache(model_path, device, batch_size)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    try:
        start_time = time.time()
        batch_logits = model_cache.get_logits(request.texts)
        
        # 确保所有logits都是浮点数列表
        for i, logits in enumerate(batch_logits):
            if not isinstance(logits, list):
                batch_logits[i] = [float(l) for l in logits]
            else:
                batch_logits[i] = [float(l) for l in logits]
        
        logger.info(f"Processed {len(request.texts)} texts in {time.time() - start_time:.2f} seconds")
        return LogitsResponse(logits=batch_logits)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/knowledge/info")
async def get_info():
    """
    获取API服务的基本信息
    """
    global model_cache
    
    info = {
        "version": "2.0.0",
        "model_path": model_path,
        "device": device if device else ("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": batch_size,
        "status": "running"
    }
    
    if model_cache:
        info["num_classes"] = model_cache.num_classes
        info["cache_size"] = len(model_cache.result_cache)
    else:
        info["model_status"] = "not_loaded"
    
    return info

@app.post("/v1/knowledge/predict", response_model=PredictResponse)
async def predict_intent(request: TextRequest):
    """
    使用预训练模型预测文本的意图分类，并对logits进行softmax归一化
    :param request: 包含文本列表的请求和可选的详细得分参数
    :return: 每个文本的预测结果
    """
    global model_cache
    
    if model_cache is None:
        try:
            model_cache = ModelCache(model_path, device, batch_size)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    try:
        start_time = time.time()
        batch_logits = model_cache.get_logits(request.texts)
        
        # 定义类别映射
        class_map = {
            0: "预约挂号",
            1: "导诊",
            2: "专家推荐",
            3: "预约管理",
            4: "咨询",
            5: "报告查询",
            6: "其他"
        }
        
        # 处理预测结果
        predictions = []
        for text, logits in zip(request.texts, batch_logits):
            # 使用torch进行softmax归一化
            logits_tensor = torch.tensor(logits)
            softmax_probs = F.softmax(logits_tensor, dim=0).numpy()
            
            # 找到最大概率的索引
            predicted_idx = softmax_probs.argmax()
            predicted_label = class_map.get(predicted_idx, "未知")
            confidence_score = float(softmax_probs[predicted_idx])
            
            # 根据参数决定是否返回all_scores
            all_scores = None
            if request.with_all_scores:
                all_scores = [
                    ScoreItem(label=class_map[i], score=float(prob)) 
                    for i, prob in enumerate(softmax_probs)
                ]
            
            predictions.append(PredictItem(
                text=text,
                label=predicted_label,
                score=confidence_score,
                all_scores=all_scores
            ))
        
        logger.info(f"Processed {len(request.texts)} texts in {time.time() - start_time:.2f} seconds")
        return PredictResponse(predictions=predictions)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting API server v2 with model: {model_path}")
    uvicorn.run(app, host="0.0.0.0", port=9000) 