#!/bin/bash
# 实验运行脚本: run_ifly25_roberta.sh

################################################################
# 实验配置
################################################################

# 实验路径
EXP_DIR="ckpt/aug"

# 模型配置
MODEL_PATH="./models/dienstag/chinese-roberta-wwm-ext"
MAX_LENGTH=256

# 数据配置
TRAIN_FILE="data/processed/train.jsonl"
VALID_FILE="data/processed/valid.jsonl"
TEST_FILE="data/processed/valid.jsonl"


################################################################
# 实验配置 1
################################################################
# 训练
EXP_NAME="ifly25_baseline_cat8k"
CKPT=$EXP_DIR/$EXP_NAME
TRAIN_FILE="data/processed/train_cat8k_splice2.jsonl"

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200



################################################################
# 实验配置 2
################################################################
# 训练
EXP_NAME="ifly25_baseline_cat8k_eda1w"
CKPT=$EXP_DIR/$EXP_NAME
TRAIN_FILE="data/processed/train_cat8k_splice2_eda1w.jsonl"

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200
 