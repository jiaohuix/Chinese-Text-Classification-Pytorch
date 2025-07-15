#!/bin/bash
# 实验运行脚本: run_roberta.sh

################################################################
# 实验配置
################################################################

# 实验路径
EXP_DIR="ckpt"

# 模型配置
MODEL_PATH="./models/dienstag/chinese-roberta-wwm-ext"
MAX_LENGTH=256

# 数据配置
TRAIN_FILE="data/thunews/train.jsonl"
VALID_FILE="data/thunews/dev.jsonl"
TEST_FILE="data/thunews/test.jsonl"



################################################################
# 实验配置 
################################################################
# 对抗训练参数
fgm_eps=1
neftune_alpha=2

EXP_NAME="fgm${fgm_eps}_neft${neftune_alpha}"
CKPT=$EXP_DIR/$EXP_NAME

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $EXP_DIR/$EXP_NAME \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 3 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 500 \
 --fgm_enabled  --fgm_eps $fgm_eps \
 --neftune_enabled --neftune_alpha $neftune_alpha

# 评估
cp $MODEL_PATH/tokenizer*  "$CKPT"
python eval.py $TEST_FILE $CKPT $CKPT/eval