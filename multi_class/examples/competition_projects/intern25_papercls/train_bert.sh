#!/bin/bash
# 实验运行脚本: run_thunews_roberta.sh

################################################################
# 实验配置
################################################################

# 实验路径
EXP_DIR="ckpt"

# 模型配置
MODEL_PATH="./models/dienstag/chinese-roberta-wwm-ext"
# MAX_LENGTH=1500
MAX_LENGTH=500
# 总样本数: 12244
# 最大长度: 1366
# 最小长度: 250
# 平均长度: 614.48

# 数据配置
TRAIN_FILE="data/papercls_1w/train_bert.jsonl"
VALID_FILE="data/papercls_1w/valid_bert.jsonl"
TEST_FILE="data/papercls_1w/valid_bert.jsonl"


################################################################
# 实验配置 1： 基线
################################################################
# 训练
EXP_NAME="papercls_baseline_len1k5"
CKPT=$EXP_DIR/$EXP_NAME

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 5 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200

# 评估
cp $MODEL_PATH/tokenizer*  "$CKPT"
python eval.py $TEST_FILE $CKPT $CKPT/eval


################################################################
# 实验配置 2： FGM
################################################################
# 对抗训练参数
fgm_eps=1
neftune_alpha=2

EXP_NAME="papercls_fgm1"
CKPT=$EXP_DIR/$EXP_NAME
bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 5 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200 \
 --fgm_enabled  --fgm_eps $fgm_eps 



################################################################
# 实验配置 3： FGM + NEFTune
################################################################
# 对抗训练参数
fgm_eps=1
neftune_alpha=1

EXP_NAME="papercls_fgm1_neft1"
CKPT=$EXP_DIR/$EXP_NAME

bash scripts/train.sh --model_path $MODEL_PATH \
 --output_dir $CKPT \
 --train_file $TRAIN_FILE \
 --valid_file $VALID_FILE \
 --test_file $TEST_FILE \
 --num_train_epochs 5 \
 --max_seq_length $MAX_LENGTH \
 --batch_size 32 --eval_steps 200  \
 --fgm_enabled  --fgm_eps $fgm_eps \
 --neftune_enabled --neftune_alpha $neftune_alpha
