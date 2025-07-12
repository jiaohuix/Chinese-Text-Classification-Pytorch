#!/usr/bin/env python
# coding: utf-8

"""
知识蒸馏模块，支持从预训练模型的logits进行蒸馏
基于transformers Trainer扩展，支持从API获取教师模型的logits
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

from transformers import Trainer, TrainingArguments
from transformers.utils import is_torch_available
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class DistillationConfig:
    """
    知识蒸馏配置类
    """
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
        api_url: str = "http://localhost:9000/v1/knowledge/logits",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        num_classes: int = 7,
    ):
        """
        参数:
            alpha: 知识蒸馏损失权重, 在[0, 1]范围内
            temperature: 蒸馏温度，用于软化logits分布
            api_url: API服务地址
            batch_size: 调用API的批次大小
            cache_dir: 缓存目录，如果使用缓存
            use_cache: 是否使用缓存保存教师模型的logits
        """
        self.alpha = alpha
        self.temperature = temperature
        self.api_url = api_url
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.num_classes = num_classes


class APILogitsClient:
    """
    调用API获取logits的客户端
    """
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.cache = {}
        self.max_cache_size = 10000  # 最大缓存条目数，防止内存占用过大
        # 创建缓存目录
        if config.use_cache and config.cache_dir:
            os.makedirs(config.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(config.cache_dir, "teacher_logits.npz")
            self._load_cache()
    
    def _load_cache(self):
        """从文件加载缓存的logits"""
        if self.config.use_cache and os.path.exists(self.cache_file):
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                texts = data['texts']
                logits = data['logits']
                
                self.cache = {}
                for i, text in enumerate(texts):
                    if i < len(logits):
                        # 转换为普通的浮点数列表
                        self.cache[str(text)] = [float(l) for l in logits[i]]
                
                logger.info(f"Loaded {len(self.cache)} cached logits from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                logger.exception(e)
    
    def _save_cache(self):
        """保存logits到缓存文件"""
        if self.config.use_cache and self.config.cache_dir:
            try:
                texts = list(self.cache.keys())
                # 确保所有 logits 都是正确的浮点数列表
                logits = []
                for text in texts:
                    item = self.cache[text]
                    # 确保是浮点数列表且长度正确
                    if not isinstance(item, list) or len(item) != self.config.num_classes:
                        item = [0.0] * self.config.num_classes
                    logits.append([float(l) for l in item])
                
                # 转换为numpy数组，显式指定dtype为float
                logits_array = np.array(logits, dtype=np.float32)
                
                np.savez(
                    self.cache_file,
                    texts=np.array(texts),
                    logits=logits_array
                )
                logger.info(f"Saved {len(self.cache)} logits to cache {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
                logger.exception(e)
    
    def prefetch_logits(self, texts_list):
        """
        预取多个batch的logits，提前获取并缓存后续batch的logits
        
        参数:
            texts_list: 多个文本batch的列表
        """
        if not self.config.use_cache:
            return
            
        # 合并并过滤掉已缓存的文本
        all_texts = []
        for texts in texts_list:
            all_texts.extend([text for text in texts if text not in self.cache])
        
        if not all_texts:
            return
            
        logger.info(f"预取 {len(all_texts)} 个文本的logits")
            
        # 分批获取logits
        for i in range(0, len(all_texts), self.config.batch_size):
            batch_texts = all_texts[i:i+self.config.batch_size]
            try:
                response = requests.post(
                    self.config.api_url,
                    json={"texts": batch_texts},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                batch_logits = response.json().get("logits", [])
                
                # 更新缓存
                for text, logits in zip(batch_texts, batch_logits):
                    self.cache[text] = logits
                    
                logger.info(f"已预取 {len(batch_texts)} 个文本的logits")
            except Exception as e:
                logger.error(f"预取API请求失败: {e}")
        
        # 管理缓存大小
        self._manage_cache_size()
        
        # 保存缓存
        self._save_cache()
    
    def _manage_cache_size(self):
        """管理缓存大小，如果超过限制则删除最早的条目"""
        if len(self.cache) > self.max_cache_size:
            # 删除20%的最早条目
            remove_count = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[key]
            logger.info(f"清理缓存: 删除了 {remove_count} 个最早的条目")
    
    def get_logits(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的logits
        如果缓存中存在，直接返回
        否则调用API获取
        
        参数:
            texts: 文本列表
            
        返回:
            logits_list: logits列表
        """
        # 检查哪些文本需要调用API
        missing_texts = [text for text in texts if text not in self.cache]
        
        if missing_texts:
            # 分批调用API
            all_api_logits = []
            for i in range(0, len(missing_texts), self.config.batch_size):
                batch_texts = missing_texts[i:i+self.config.batch_size]
                try:
                    response = requests.post(
                        self.config.api_url,
                        json={"texts": batch_texts},
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    response.raise_for_status()
                    batch_logits = response.json().get("logits", [])
                    
                    # 确保每个logits都是列表并且包含浮点数
                    for j, logits in enumerate(batch_logits):
                        if not isinstance(logits, list):
                            logger.warning(f"API返回了非列表格式的logits: {type(logits)}")
                            logits = [0.0] * self.config.num_classes
                        elif len(logits) != self.config.num_classes:
                            logger.warning(f"API返回的logits长度不匹配: {len(logits)} vs {self.config.num_classes}")
                            # 长度调整
                            if len(logits) < self.config.num_classes:
                                logits = logits + [0.0] * (self.config.num_classes - len(logits))
                            else:
                                logits = logits[:self.config.num_classes]
                        
                        # 转换为浮点数
                        batch_logits[j] = [float(l) for l in logits]
                    
                    all_api_logits.extend(batch_logits)
                    
                    # 更新缓存
                    for text, logits in zip(batch_texts, batch_logits):
                        self.cache[text] = logits
                        
                except Exception as e:
                    logger.error(f"API request failed: {e}")
                    # 如果API调用失败，返回零向量
                    zero_logits = [[0.0] * self.config.num_classes for _ in range(len(batch_texts))]
                    all_api_logits.extend(zero_logits)
            
            # 保存更新后的缓存
            if self.config.use_cache:
                self._save_cache()
                # 管理缓存大小
                self._manage_cache_size()
        
        # 返回所有文本的logits，并确保格式一致
        result = []
        for text in texts:
            if text in self.cache:
                logits = self.cache[text]
                # 进行最后的验证和转换
                if not isinstance(logits, list) or len(logits) != self.config.num_classes:
                    logits = [0.0] * self.config.num_classes
                # 确保所有值都是浮点数
                result.append([float(l) for l in logits])
            else:
                result.append([0.0] * self.config.num_classes)
        
        return result


class DistillationTrainer(Trainer):
    """
    知识蒸馏训练器
    从API获取教师模型的logits，用于训练学生模型
    """
    def __init__(
        self,
        distill_config: DistillationConfig,
        tokenizer=None,
        **kwargs
    ):
        """
        参数:
            distill_config: 知识蒸馏配置
            其他参数与Trainer相同
        """
        super().__init__(**kwargs)
        self.distill_config = distill_config
        self.api_client = APILogitsClient(distill_config)
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算蒸馏损失
        1. 计算学生模型的常规交叉熵损失
        2. 从API获取教师模型的logits
        3. 计算学生模型与教师模型的KL散度损失
        4. 组合两种损失
        
        参数:
            model: 学生模型
            inputs: 模型输入
            return_outputs: 是否返回模型输出
            
        返回:
            loss: 总损失
            outputs: 如果return_outputs=True，返回模型输出
        """
        # 学生模型前向传播
        outputs = model(**inputs)
        student_loss = outputs.loss
        
        # 获取教师模型的logits
        if "sentence" in inputs:
            # 如果inputs已经包含原始文本
            texts = inputs["sentence"]
        else:
            # 从input_ids重建文本
            texts = self._decode_input_ids(inputs["input_ids"])
        
        # 获取教师模型的logits
        teacher_logits_list = self.api_client.get_logits(texts)
        
        # 确保所有logits都是规范的浮点数列表，并且长度一致
        try:
            # 验证数据格式并进行处理
            processed_logits = []
            expected_length = self.distill_config.num_classes
            
            for i, logits in enumerate(teacher_logits_list):
                # 检查每个logits是否是列表或数组
                if not isinstance(logits, (list, np.ndarray)):
                    logger.warning(f"Logits at index {i} is not a list or array, using zeros instead")
                    logits = [0.0] * expected_length
                
                # 确保logits长度正确
                if len(logits) != expected_length:
                    logger.warning(f"Logits at index {i} has incorrect length {len(logits)}, expected {expected_length}")
                    # 如果长度不足，填充零；如果长度过长，截断
                    if len(logits) < expected_length:
                        logits = logits + [0.0] * (expected_length - len(logits))
                    else:
                        logits = logits[:expected_length]
                
                # 确保所有值都是浮点数
                processed_logits.append([float(v) for v in logits])
            
            # 转换为PyTorch张量
            teacher_logits = torch.tensor(processed_logits, dtype=torch.float, device=outputs.logits.device)
            
        except Exception as e:
            logger.error(f"Error processing teacher logits: {e}")
            # 出错时使用零向量
            teacher_logits = torch.zeros_like(outputs.logits)
        
        # 计算蒸馏损失 (KL散度)
        distill_loss = self._compute_distillation_loss(
            outputs.logits, 
            teacher_logits, 
            self.distill_config.temperature
        )
        
        # 组合损失
        # (1 - alpha) * CE + alpha * KL
        loss = (1 - self.distill_config.alpha) * student_loss + self.distill_config.alpha * distill_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        计算知识蒸馏损失 (KL散度)
        
        参数:
            student_logits: 学生模型的logits
            teacher_logits: 教师模型的logits
            temperature: 温度参数
            
        返回:
            kl_loss: KL散度损失
        """
        # 应用温度
        student_logits_temp = student_logits / temperature
        teacher_logits_temp = teacher_logits / temperature
        
        # 计算KL散度
        kl_loss = F.kl_div(
            F.log_softmax(student_logits_temp, dim=-1),
            F.softmax(teacher_logits_temp, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)  # 乘以温度的平方进行缩放
        
        return kl_loss
    
    def _decode_input_ids(self, input_ids):
        """
        将input_ids解码为文本
        
        参数:
            input_ids: 输入的token IDs
            
        返回:
            texts: 解码后的文本列表
        """
        if self.tokenizer is None:
            logger.warning("Tokenizer not provided, cannot decode input_ids")
            return [""] * len(input_ids)
        
        texts = []
        for ids in input_ids:
            # 移除特殊token (如padding, CLS, SEP等)
            # 过滤掉padding token (通常是0)
            filtered_ids = ids[ids != 0].cpu().numpy()
            text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
            texts.append(text)
        
        return texts
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        重写_save方法，确保保存模型时同时保存tokenizer
        """
        # 首先调用父类的_save方法保存模型
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用父类的保存方法
        super()._save(output_dir, state_dict)
        
        # 额外保存tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        重写save_model方法，确保保存完整的模型和tokenizer
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用父类的保存方法
        super().save_model(output_dir, _internal_call)
        
        # 确保tokenizer保存
        if self.tokenizer is not None and not _internal_call:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")

    def get_train_dataloader(self) -> DataLoader:
        """
        重写获取训练数据加载器的方法，添加预取功能
        """
        dataloader = super().get_train_dataloader()
        
        # 如果配置了预取，则启动预取过程
        if self.distill_config.use_cache:
            try:
                # 尝试预取前几个batch的logits
                prefetch_batches = []
                prefetch_count = 0
                max_prefetch = 5  # 最多预取5个batch
                
                # 从dataloader中获取几个batch用于预取
                dataloader_iter = iter(dataloader)
                for _ in range(max_prefetch):
                    try:
                        batch = next(dataloader_iter)
                        if "sentence" in batch:
                            texts = batch["sentence"]
                        elif "input_ids" in batch:
                            texts = self._decode_input_ids(batch["input_ids"])
                        else:
                            continue
                        
                        prefetch_batches.append(texts)
                        prefetch_count += 1
                    except StopIteration:
                        break
                
                # 启动预取
                if prefetch_batches:
                    logger.info(f"预取 {len(prefetch_batches)} 个batch的logits")
                    self.api_client.prefetch_logits(prefetch_batches)
            except Exception as e:
                logger.warning(f"Logits预取失败: {e}")
        
        return dataloader


# 知识蒸馏训练参数
@dataclass
class DistillationTrainingArguments(TrainingArguments):
    """
    扩展TrainingArguments，添加知识蒸馏相关参数
    """
    # 知识蒸馏参数
    distill_alpha: float = field(
        default=0.5,
        metadata={"help": "知识蒸馏损失的权重，介于0和1之间"}
    )
    distill_temperature: float = field(
        default=2.0,
        metadata={"help": "知识蒸馏的温度参数"}
    )
    teacher_api_url: str = field(
        default="http://localhost:9000/v1/knowledge/logits",
        metadata={"help": "教师模型API的URL"}
    )
    api_batch_size: int = field(
        default=32,
        metadata={"help": "调用API时的批次大小"}
    )
    use_logits_cache: bool = field(
        default=True,
        metadata={"help": "是否缓存教师模型的logits"}
    )
    logits_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "教师模型logits的缓存目录"}
    )
    num_classes: int = field(
        default=7,
        metadata={"help": "分类任务的类别数量"}
    )


# 在主脚本中使用的工厂函数
def get_distillation_trainer(
    model, 
    args, 
    tokenizer=None, 
    train_dataset=None, 
    eval_dataset=None, 
    compute_metrics=None, 
    data_collator=None
):
    """
    创建DistillationTrainer实例
    
    参数:
        model: 学生模型
        args: DistillationTrainingArguments
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        compute_metrics: 计算指标的函数
        data_collator: 数据整理函数
        
    返回:
        trainer: DistillationTrainer实例
    """
    # 创建知识蒸馏配置
    distill_config = DistillationConfig(
        alpha=args.distill_alpha,
        temperature=args.distill_temperature,
        api_url=args.teacher_api_url,
        batch_size=args.api_batch_size,
        cache_dir=args.logits_cache_dir,
        use_cache=args.use_logits_cache,
        num_classes=args.num_classes
    )
    
    # 创建训练器
    trainer = DistillationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        distill_config=distill_config
    )
    
    return trainer 