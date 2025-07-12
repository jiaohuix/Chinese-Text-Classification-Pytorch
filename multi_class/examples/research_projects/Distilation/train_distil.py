#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
'''
bug:
1 rdrop kl loss 需要sum而不是mean
2 rdropfgm trainner参数没传进去，get_trainer
'''
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import matplotlib.pyplot as plt
from transformers import TrainerCallback

# 导入知识蒸馏相关模块
from distillation import DistillationTrainer, DistillationConfig, get_distillation_trainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.52.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)




# @dataclass
# class RDropTrainingArguments(TrainingArguments):
#     """
#     扩展了TrainingArguments，添加了R-Drop相关参数
#     """
#     rdrop_alpha: Optional[float] = field(
#         default=0.0,
#         metadata={"help": "R-Drop KL散度损失的权重系数，设为0则禁用R-Drop"}
#     )
#     # rdrop_enabled: Optional[bool] = field(
#     #     default=False,
#     #     metadata={"help": "是否启用R-Drop正则化"}
#     # )

#     plot_loss: Optional[bool] = field(
#         default=False,
#         metadata={"help": "是否画loss曲线"}
#     )


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    """
    扩展了TrainingArguments，添加多种训练增强相关参数
    """
    # R-Drop相关参数
    rdrop_alpha: Optional[float] = field(
        default=0.0,
        metadata={"help": "R-Drop KL散度损失的权重系数，设为0则禁用R-Drop"}
    )
    
    # FGM相关参数
    fgm_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": "是否启用FGM对抗训练"}
    )
    fgm_epsilon: Optional[float] = field(
        default=1.0,
        metadata={"help": "FGM扰动大小"}
    )
    fgm_emb_name: Optional[str] = field(
        default="word_embeddings",
        metadata={"help": "FGM扰动的embedding层名称"}
    )
    
    # 知识蒸馏相关参数
    distill_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": "是否启用知识蒸馏"}
    )
    distill_alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "知识蒸馏损失的权重，介于0和1之间"}
    )
    distill_temperature: Optional[float] = field(
        default=2.0,
        metadata={"help": "知识蒸馏的温度参数"}
    )
    teacher_api_url: Optional[str] = field(
        default="http://localhost:9000/v1/knowledge/logits",
        metadata={"help": "教师模型API的URL"}
    )
    api_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "调用API时的批次大小"}
    )
    use_logits_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "是否缓存教师模型的logits"}
    )
    logits_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "教师模型logits的缓存目录"}
    )
    num_classes: Optional[int] = field(
        default=7,
        metadata={"help": "分类任务的类别数量"}
    )
    
    # 可视化参数
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "是否画loss曲线"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json","jsonl"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert validation_extension == train_extension, (
                "`validation_file` should have the same extension (csv or json) as `train_file`."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


# def compute_kl_loss(p, q):
#     """计算两个分布之间的KL散度"""
#     p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
#     q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
#     # 求和并平均
#     p_loss = p_loss.sum(-1)
#     q_loss = q_loss.sum(-1)
#     loss = (p_loss + q_loss) / 2
#     return loss.mean()


def compute_kl_loss(p, q, pad_mask=None):
    """计算两个分布之间的KL散度"""
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class RDropTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 第一次前向传播
        outputs_1 = model(**inputs)
        logits_1 = outputs_1.logits
        loss_1 = outputs_1.loss
        
        # 第二次前向传播
        outputs_2 = model(**inputs)
        logits_2 = outputs_2.logits
        loss_2 = outputs_2.loss
        
        # 计算KL散度损失
        kl_loss = compute_kl_loss(logits_1, logits_2)
        
        # 最终损失
        loss = (loss_1 + loss_2) / 2 + self.args.rdrop_alpha * kl_loss
        
        return (loss, outputs_1) if return_outputs else loss

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




class FGM:
    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
        self._is_attacked = False  # 添加状态跟踪

    def attack(self):
        """添加对抗扰动"""
        if self._is_attacked:
            raise RuntimeError("Already attacked! Call restore() first")
            
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm > 0:  # 更安全的判断
                    r_at = self.epsilon * param.grad / (norm + 1e-6)  # 防止除零
                    param.data.add_(r_at)
        self._is_attacked = True

    def restore(self):
        """恢复原始embedding"""
        if not self._is_attacked:
            return
            
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
        self._is_attacked = False



# 在Trainer中使用FGM
# class FGMTrainer(Trainer):
#     def training_step(self, model, inputs):
#         # 正常前向传播和反向传播
#         loss = super().training_step(model, inputs)
        
#         # 创建FGM对象
#         fgm = FGM(model, epsilon=self.args.fgm_epsilon, emb_name=self.args.fgm_emb_name)

#         # 对抗攻击
#         fgm.attack()
        
#         # 对抗样本的损失计算
#         adv_inputs = {k: v for k, v in inputs.items()}
#         adv_loss = self.compute_loss(model, adv_inputs)
        
#         # 反向传播对抗损失
#         adv_loss.backward()
        
#         # 恢复嵌入
#         fgm.restore()
        
#         return loss

class FGMTrainer(Trainer):
    def training_step(self, model, inputs):
        # 原始训练步骤（自动包含AMP上下文）
        loss = super().training_step(model, inputs)  # 原始loss

        # logger.info(f"self.args.fp16 {self.args.fp16}")
        # logger.info(f"self.use_apex {self.use_apex}")
        # logger.info(f"self.self.args.fgm_enabled {self.args.fgm_enabled}")
        
        
        if self.args.fgm_enabled:
            # FGM对抗扰动
            fgm = FGM(model)
            fgm.attack()  # 在embedding上添加扰动
            
            # 复用父类的training_step计算对抗loss（自动处理AMP/梯度累积等）
            adv_loss = super().training_step(model, inputs)  # 对抗样本loss
            
            fgm.restore()  # 恢复原始参数
            # print("loss:", float(loss), "adv_loss:", float(adv_loss))   # True 表示使用 Apex AMP

            # logger.info(f"loss: {float(loss)} | adv_loss: {float(adv_loss)}")
            
            # 合并损失（原始loss + 对抗loss）
            loss = (loss + adv_loss) / 2
        
        return loss  # 父类会自动处理backward


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




class RDropFGMTrainer(Trainer):
    """
    结合RDrop和FGM的Trainer类
    根据论文: R-Drop: Regularized Dropout for Neural Networks
    """
    def __init__(self, rdrop_alpha=1.0, epsilon=1.0, emb_name='word_embeddings', **kwargs):
        super().__init__(**kwargs)
        self.rdrop_alpha = rdrop_alpha
        logger.info(f"Using RDropFGMTrainer with rdrop_alpha={rdrop_alpha} in cls")
        self.fgm = FGM(self.model, epsilon=epsilon, emb_name=emb_name)
    
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     重写计算损失函数
    #     1. 常规前向传播获得第一组logits
    #     2. 使用FGM添加扰动
    #     3. 对抗样本前向传播获得第二组logits
    #     4. 计算交叉熵损失和KL散度损失
    #     """
    #     # 第一次前向传播（正常样本）
    #     outputs_1 = model(**inputs)
    #     loss_1 = outputs_1.loss
        
    #     # 应用FGM扰动
    #     self.fgm.attack()
        
    #     # 第二次前向传播（对抗样本）
    #     outputs_2 = model(**inputs)
    #     loss_2 = outputs_2.loss
        
    #     # 恢复embedding
    #     self.fgm.restore()
        
    #     # 计算KL散度 (修改：不使用detach，使两个方向都有梯度)
    #     loss_kl = compute_kl_loss(outputs_1.logits, outputs_2.logits)
        
    #     # 合并损失 (交叉熵损失 + KL散度损失)
    #     loss = 0.5 * (loss_1 + loss_2) + self.rdrop_alpha * loss_kl

    #     print(f"Using rdrop_alpha: {self.rdrop_alpha}")
    #     print(f"CE loss: {0.5 * (loss_1 + loss_2)}, KL loss: {loss_kl}, KL contribution: {self.rdrop_alpha * loss_kl}")


    #     return (loss, outputs_1) if return_outputs else loss

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写计算损失函数
        1. 常规前向传播获得第一组logits
        2. 使用FGM添加扰动
        3. 对抗样本前向传播获得第二组logits
        4. 计算交叉熵损失和KL散度损失
        """
        
        # 第一次前向传播（正常样本）
        outputs_1 = model(**inputs)
        loss_1 = outputs_1.loss
        
        # 应用FGM扰动
        self.fgm.attack()
        
        # 第二次前向传播（对抗样本）
        outputs_2 = model(**inputs)
        loss_2 = outputs_2.loss
        
        # 恢复embedding
        self.fgm.restore()
        
        # 计算KL散度 (修改：不使用detach，使两个方向都有梯度)
        loss_kl = compute_kl_loss(outputs_1.logits, outputs_2.logits)
        
        # 合并损失 (交叉熵损失 + KL散度损失)
        ce_loss = 0.5 * (loss_1 + loss_2)
        kl_contribution = self.rdrop_alpha * loss_kl
        loss = ce_loss + kl_contribution

        # 使用logging而不是print，避免干扰训练输出
        # logger.info(f"Using rdrop_alpha: {self.rdrop_alpha}")
        # logger.info(f"CE loss: {ce_loss.item():.4f}, KL loss: {loss_kl.item():.4f}, KL contribution: {kl_contribution.item():.4f}")
    
        return (loss, outputs_1) if return_outputs else loss

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


class LossCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs and state.is_local_process_zero:
                self.train_losses.append(logs['loss'])
                self.train_steps.append(state.global_step)
            
            if 'eval_loss' in logs and state.is_local_process_zero:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_steps, self.train_losses, label='Training Loss')
            
            if self.eval_losses:
                plt.plot(self.eval_steps, self.eval_losses, label='Validation Loss')
            
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/loss_curve.png')
            
            # 保存数据到文件中，以便后续使用
            np.save(f'{self.output_dir}/train_losses.npy', np.array(self.train_losses))
            np.save(f'{self.output_dir}/train_steps.npy', np.array(self.train_steps))
            np.save(f'{self.output_dir}/eval_losses.npy', np.array(self.eval_losses))
            np.save(f'{self.output_dir}/eval_steps.npy', np.array(self.eval_steps))


def get_label_list(raw_dataset, split="train") -> list[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


# 根据参数选择相应的Trainer
def get_trainer(model, args, train_dataset=None, eval_dataset=None, 
                compute_metrics=None, data_collator=None, tokenizer=None):
    
    use_rdrop = args.rdrop_alpha > 0
    use_fgm = args.fgm_enabled
    use_distill = args.distill_enabled
    
    if use_distill:
        logger.info(f"Using DistillationTrainer with alpha={args.distill_alpha}, temperature={args.distill_temperature}")
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
        
        # 创建蒸馏训练器
        trainer = DistillationTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer,
            distill_config=distill_config
        )
    elif use_rdrop and use_fgm:
        logger.info(f"Using RDropFGMTrainer with rdrop_alpha={args.rdrop_alpha}, fgm_epsilon={args.fgm_epsilon}")
        trainer = RDropFGMTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            rdrop_alpha=args.rdrop_alpha, 
            epsilon=args.fgm_epsilon, 
            emb_name=args.fgm_emb_name,
        )
    elif use_rdrop:
        logger.info(f"Using RDropTrainer with rdrop_alpha={args.rdrop_alpha}")
        trainer = RDropTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    elif use_fgm:
        logger.info(f"Using FGMTrainer with fgm_epsilon={args.fgm_epsilon}")
        trainer = FGMTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    else:
        logger.info("Using standard Trainer")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    
    return trainer


def natural_sort_key(s):
    try:
        return (0, int(s))  # 数字字符串，按数值排序
    except ValueError:
        return (1, s)       # 非数字字符串，按字典序排序


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EnhancedTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert test_extension == train_extension, (
                    "`test_file` should have the same extension (csv or json) as `train_file`."
                )
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split] = raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Trying to have good defaults here, don't hesitate to tweak to your needs.

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regression requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_list = sorted(label_list, key=natural_sort_key)
        print("label_list",label_list)

        label_to_id = {v: i for i, v in enumerate(label_list)}
        print("label_to_id",label_to_id)
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info(f"label2id: {model.config.label2id}")
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None
    print("label_to_id",label_to_id)
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: list[str]) -> list[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[str(label)]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        else:
            # Check if "sentence" field exists, otherwise use "text" field
            if "sentence" not in examples and "text" in examples:
                examples["sentence"] = examples["text"]
                
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel", cache_dir=model_args.cache_dir)
            if is_multi_label
            else evaluate.load(data_args.metric_name, cache_dir=model_args.cache_dir)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
                logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
        else:
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    # if training_args.rdrop_alpha > 0:
    #     logger.info(f"Using Rdrop Trainer with alpha: {training_args.rdrop_alpha}.")

    #     trainer = RDropTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset if training_args.do_train else None,
    #         eval_dataset=eval_dataset if training_args.do_eval else None,
    #         compute_metrics=compute_metrics,
    #         # processing_class=tokenizer,
    #         data_collator=data_collator,
    #     )
    # else:
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset if training_args.do_train else None,
    #         eval_dataset=eval_dataset if training_args.do_eval else None,
    #         compute_metrics=compute_metrics,
    #         # processing_class=tokenizer,
    #         data_collator=data_collator,
    #     )
    # 初始化trainer
    trainer = get_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,  # 传递tokenizer给get_trainer
    )

    print("model",model)

    # 添加回调函数
    if training_args.plot_loss:
        loss_callback = LossCallback(training_args.output_dir)
        trainer.add_callback(loss_callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        if is_regression:
            predictions = np.squeeze(predictions)
        elif is_multi_label:
            # Convert logits to multi-hot encoding. We compare the logits to 0 instead of 0.5, because the sigmoid is not applied.
            # You can also pass `preprocess_logits_for_metrics=lambda logits, labels: nn.functional.sigmoid(logits)` to the Trainer
            # and set p > 0.5 below (less efficient in this case)
            predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
        else:
            predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    elif is_multi_label:
                        # recover from multi-hot encoding
                        item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                        writer.write(f"{index}\t{item}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
        logger.info(f"Predict results saved at {output_predict_file}")
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
