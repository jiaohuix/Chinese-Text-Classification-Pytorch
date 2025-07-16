#!/bin/bash
# 实验运行脚本: run_ifly25_roberta.sh

################################################################
# 实验配置
################################################################

# 实验路径
EXP_DIR="ckpt/models"

# 模型配置
MAX_LENGTH=256

# 数据配置
TRAIN_FILE="data/processed/train.jsonl"
VALID_FILE="data/processed/valid.jsonl"
TEST_FILE="data/processed/valid.jsonl"


# ################################################################
# # 实验配置 1
# ################################################################
# # 训练
# EXP_NAME="ifly25_roberta"
# CKPT=$EXP_DIR/$EXP_NAME
# MODEL_PATH="./models/dienstag/chinese-roberta-wwm-ext"

# bash scripts/train.sh --model_path $MODEL_PATH \
#  --output_dir $CKPT \
#  --train_file $TRAIN_FILE \
#  --valid_file $VALID_FILE \
#  --test_file $TEST_FILE \
#  --num_train_epochs 3 \
#  --max_seq_length $MAX_LENGTH \
#  --batch_size 32 --eval_steps 200


################################################################
# 实验配置 2
################################################################
# 训练
EXP_NAME="ifly25_nlp_roberta_backbone_lite_std"
CKPT=$EXP_DIR/$EXP_NAME
MODEL_PATH="./models/nlp_roberta_backbone_lite_std"

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200

################################################################
# 实验配置 3
################################################################
# 训练
EXP_NAME="ifly25_nlp_roberta_backbone_base_std"
CKPT=$EXP_DIR/$EXP_NAME
MODEL_PATH="./models/nlp_roberta_backbone_base_std"

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200



################################################################
# 实验配置 4
################################################################
# 训练
EXP_NAME="ifly25_bge_small15"
CKPT=$EXP_DIR/$EXP_NAME
MODEL_PATH="./models/bge-small-zh-v1.5"

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200


################################################################
# 实验配置 5
################################################################
# 训练
EXP_NAME="ifly25_bgem3"
CKPT=$EXP_DIR/$EXP_NAME
MODEL_PATH="./models/bge-m3"

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200

 
