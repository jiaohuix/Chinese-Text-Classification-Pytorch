#!/bin/bash
# 实验运行脚本: run_ifly25_roberta.sh

################################################################
# 实验配置
################################################################

# 实验路径
EXP_DIR="ckpt"

# 模型配置
MODEL_PATH="./models/dienstag/chinese-roberta-wwm-ext"
MAX_LENGTH=256

# 数据配置
TRAIN_FILE="data/kfolds/fold_1/train.jsonl"
VALID_FILE="data/kfolds/fold_1/dev.jsonl"
TEST_FILE="data/kfolds/fold_1/test.jsonl"
TEST_FILE="data/kfolds/fold_1/dev.jsonl"


################################################################
# 实验配置 
################################################################
# 训练
EXP_NAME="ifly25_baseline"
CKPT=$EXP_DIR/$EXP_NAME

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 500

# 评估
cp $MODEL_PATH/tokenizer*  "$CKPT"
python eval.py $TEST_FILE $CKPT $CKPT/eval