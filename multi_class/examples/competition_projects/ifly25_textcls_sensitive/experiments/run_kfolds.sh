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




################################################################
# 实验: kfold1
################################################################
# 数据配置
TRAIN_FILE="data/kfolds/fold_1/train.jsonl"
VALID_FILE="data/kfolds/fold_1/dev.jsonl"
TEST_FILE="data/kfolds/fold_1/test.jsonl"
TEST_FILE="data/kfolds/fold_1/dev.jsonl"

EXP_NAME="ifly25_baseline_fold1"
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


################################################################
# 实验: kfold2
################################################################
# 数据配置
TRAIN_FILE="data/kfolds/fold_2/train.jsonl"
VALID_FILE="data/kfolds/fold_2/dev.jsonl"
TEST_FILE="data/kfolds/fold_2/test.jsonl"
TEST_FILE="data/kfolds/fold_2/dev.jsonl"

EXP_NAME="ifly25_baseline_fold2"
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



################################################################
# 实验: kfold3
################################################################
# 数据配置
TRAIN_FILE="data/kfolds/fold_3/train.jsonl"
VALID_FILE="data/kfolds/fold_3/dev.jsonl"
TEST_FILE="data/kfolds/fold_3/test.jsonl"
TEST_FILE="data/kfolds/fold_3/dev.jsonl"

EXP_NAME="ifly25_baseline_fold3"
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


################################################################
# 实验: kfold4
################################################################
# 数据配置
TRAIN_FILE="data/kfolds/fold_4/train.jsonl"
VALID_FILE="data/kfolds/fold_4/dev.jsonl"
TEST_FILE="data/kfolds/fold_4/test.jsonl"
TEST_FILE="data/kfolds/fold_4/dev.jsonl"

EXP_NAME="ifly25_baseline_fold4"
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


################################################################
# 实验: kfold5
################################################################
# 数据配置
TRAIN_FILE="data/kfolds/fold_5/train.jsonl"
VALID_FILE="data/kfolds/fold_5/dev.jsonl"
TEST_FILE="data/kfolds/fold_5/test.jsonl"
TEST_FILE="data/kfolds/fold_5/dev.jsonl"

EXP_NAME="ifly25_baseline_fold5"
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