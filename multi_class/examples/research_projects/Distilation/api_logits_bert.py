'''
提供基于instruct训练的Qwen3模型的蒸馏logits服务。
输入：texts list[text]
输出：logits list[list[float]]
基于fastapi实现

v0.1.0 随机生成float16的logits
v0.2.0 用训练好的bert的logits提供服务
v0.3.0 用裁剪词表的llm提供logits蒸馏服务
v4.0.0 用完整词表的llm并直接计算logits提供服务 (当前版本)

python scripts_unsloth/api_logits_v4.py outputs_qwen3_instruct/checkpoint-1400

'''

import os
import time
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
import logging
from tqdm import tqdm
import torch.nn.functional as F
import sys

# FastAPI相关
from fastapi import FastAPI, HTTPException
import uvicorn

# 大模型相关
from unsloth import FastLanguageModel
from peft import PeftModel

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

# 大模型缓存
class LLMCache:
    def __init__(
        self, 
        model_name: str, 
        lora_path: str, 
        device: str = None, 
        batch_size: int = 8,
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
        num_classes: int = 7
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        
        self.model = None
        self.tokenizer = None
        self.number_token_ids = []  # 存储数字token的IDs
        self.result_cache = {}  # 结果缓存
        self.max_cache_size = 10000  # 最大缓存条目数
        
        self.load_model()
        
    def load_model(self):
        """加载模型、分词器和配置"""
        logger.info(f"Loading model: {self.model_name} with LoRA: {self.lora_path}")
        start_time = time.time()
        
        try:
            # 1. 加载基础模型和tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                load_in_4bit=self.load_in_4bit,
                max_seq_length=self.max_seq_length,
                dtype=None  # 自动检测
            )
            
            # 2. 获取数字token IDs (用于约束生成和提取logits)
            for i in range(0, self.num_classes):
                token_id = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
                self.number_token_ids.append(token_id)
            logger.info(f"Number token IDs: {self.number_token_ids}")
            
            # 3. 加载LoRA适配器
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            logger.info("Loaded LoRA adapter")
            
            # 4. 合并LoRA权重
            self.model = self.model.merge_and_unload()
            logger.info("Merged LoRA weights")
            
            # 5. 设置为推理模式
            FastLanguageModel.for_inference(self.model)
            logger.info("Model set to inference mode")
            
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_logits(self, texts: List[str]) -> List[List[float]]:
        """获取文本的logits，支持批处理"""
        # 准备结果和需要处理的文本列表
        result = [None] * len(texts)
        texts_to_process = []
        indices_to_process = []
        
        # 一次性检查所有文本是否在缓存中
        for i, text in enumerate(texts):
            if text in self.result_cache:
                result[i] = self.result_cache[text]
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        # 如果所有文本都在缓存中，直接返回
        if not texts_to_process:
            logger.info(f"All {len(texts)} texts found in cache")
            return result
        
        logger.info(f"Processing {len(texts_to_process)} texts in batches")
        
        # 分批处理未缓存的文本
        for i in range(0, len(texts_to_process), self.batch_size):
            batch_texts = texts_to_process[i:i+self.batch_size]
            batch_indices = indices_to_process[i:i+self.batch_size]
            
            # 真正并行处理批量文本
            batch_logits = self._process_batch(batch_texts)
            
            # 更新结果和缓存
            for j, (text, logits, original_idx) in enumerate(zip(batch_texts, batch_logits, batch_indices)):
                result[original_idx] = logits
                self.result_cache[text] = logits
        
        # 管理缓存大小
        self._manage_cache_size()
        
        return result
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """处理一批文本，真正并行地返回logits"""
        if not texts:
            return []
        
        # 1. 将所有文本转换为对话格式
        conversations = [self._format_conversation(text) for text in texts]
        
        # 2. 批量应用chat模板
        prompts = [
            self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            ) for conv in conversations
        ]
        
        # 3. 批量编码输入
        # 使用padding=True来处理不同长度的输入
        batch_inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.model.device)
        
        # 4. 一次性计算所有文本的logits
        with torch.inference_mode():
            outputs = self.model(**batch_inputs)
            batch_logits = outputs.logits
            
            # 创建结果列表
            results = []
            
            # 5. 对每个样本，提取最后一个非padding token的logits
            for i, input_ids in enumerate(batch_inputs.input_ids):
                # 获取该样本的有效长度（非padding部分）
                valid_length = torch.sum(batch_inputs.attention_mask[i])
                
                # 获取最后一个token位置的logits
                last_token_logits = batch_logits[i, valid_length - 1]
                
                # 只考虑数字token的logits
                class_logits = last_token_logits[self.number_token_ids].cpu().numpy().tolist()
                results.append(class_logits)
        
        return results
    
    def _format_conversation(self, text: str) -> List[Dict[str, str]]:
        """将文本转换为对话格式"""
        system_prompt = "你是一个医疗场景下的意图识别助手，你的任务是帮助用户识别医疗问题的意图类别。"
        
        class_description = """
class 0: 预约挂号
class 1: 导诊
class 2: 专家推荐
class 3: 预约管理
class 4: 咨询
class 5: 报告查询
class 6: 其他
"""
        
        user_message = f"""这是一条用户查询，请做医疗场景的意图识别:
===
{text}
===

把这条查询分类为以下意图类型之一：
{class_description}

请仅回答类别编号。"""

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return conversation
    
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
    title="Knowledge Distillation API v4",
    description="提供基于Instruct训练的大模型的logits蒸馏服务",
    version="4.0.0"
)

# 全局变量
llm_cache = None

# 从环境变量获取配置
model_name = os.environ.get("MODEL_NAME", "/home/wonders/zjh/Projects/pretrained_models/Qwen3-4B")
lora_path = os.environ.get("LORA_PATH", "outputs_qwen3_instruct/checkpoint-1400")
# print(f"lora_path: {lora_path}")
logger.info(f"lora_path: {lora_path}")
device = os.environ.get("DEVICE", None)  # 自动选择
batch_size = int(os.environ.get("BATCH_SIZE", 8))
num_classes = int(os.environ.get("NUM_CLASSES", 7))
load_in_4bit = os.environ.get("LOAD_IN_4BIT", "True").lower() == "true"
max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", 1024))

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global llm_cache
    try:
        llm_cache = LLMCache(
            model_name=model_name,
            lora_path=lora_path,
            device=device,
            batch_size=batch_size,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            num_classes=num_classes
        )
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {str(e)}")
        # 应用将继续运行，但API调用会失败

@app.post("/v1/knowledge/logits", response_model=LogitsResponse)
async def distill_logits(request: TextRequest):
    """
    使用大模型为输入的文本列表生成logits
    :param request: 包含文本列表的请求
    :return: 每个文本对应的logits列表
    """
    global llm_cache
    
    if llm_cache is None:
        try:
            llm_cache = LLMCache(
                model_name=model_name,
                lora_path=lora_path,
                device=device,
                batch_size=batch_size,
                load_in_4bit=load_in_4bit,
                max_seq_length=max_seq_length,
                num_classes=num_classes
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    try:
        start_time = time.time()
        batch_logits = llm_cache.get_logits(request.texts)
        
        # 确保所有logits都是浮点数列表
        for i, logits in enumerate(batch_logits):
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
    global llm_cache
    
    info = {
        "version": "4.0.0",
        "model_name": model_name,
        "lora_path": lora_path,
        "device": device if device else ("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": batch_size,
        "num_classes": num_classes,
        "load_in_4bit": load_in_4bit,
        "max_seq_length": max_seq_length,
        "status": "running",
        "implementation": "direct_logits"  # 标识使用直接计算logits的方法
    }
    
    if llm_cache:
        info["cache_size"] = len(llm_cache.result_cache)
    else:
        info["model_status"] = "not_loaded"
    
    return info

@app.post("/v1/knowledge/predict", response_model=PredictResponse)
async def predict_intent(request: TextRequest):
    """
    使用大模型预测文本的意图分类，并对logits进行softmax归一化
    :param request: 包含文本列表的请求和可选的详细得分参数
    :return: 每个文本的预测结果
    """
    global llm_cache
    
    if llm_cache is None:
        try:
            llm_cache = LLMCache(
                model_name=model_name,
                lora_path=lora_path,
                device=device,
                batch_size=batch_size,
                load_in_4bit=load_in_4bit,
                max_seq_length=max_seq_length,
                num_classes=num_classes
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    try:
        start_time = time.time()
        batch_logits = llm_cache.get_logits(request.texts)
        
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
    # 支持命令行参数指定模型路径
    if len(sys.argv) > 1:
        lora_path = sys.argv[1]
        logger.info(f"Using LoRA path from command line: {lora_path}")
    
    print(f"Starting API server v4 with model: {model_name} and LoRA: {lora_path}")
    print(f"Number of classes: {num_classes}")
    uvicorn.run(app, host="0.0.0.0", port=9000) 