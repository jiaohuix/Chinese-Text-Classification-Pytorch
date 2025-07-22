#!/bin/bash

MODEL_PATH="/share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat"

# day0720 周日 从多类别采样数据，确保v类存在
# neftune参数
export USE_NEFTUNE=true
export NEFTUNE_ALPHA=1
DATA_PATH="data/mclass/papercls26_v21/train.jsonl"
CKPT="./swift_output/InternLM2.5-1.8B-Lora-1w3v21-neft1"
mkdir -p $CKPT


# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/internlm2.5-1.8b_lora_sft_${TIMESTAMP}.log"
echo "start training..." >> $LOG_FILE
# 设置CUDA设备
#export NPROC_PER_NODE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

nohup swift sft \
    --model $MODEL_PATH \
    --train_type lora \
    --dataset $DATA_PATH \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --split_dataset_ratio 0 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --save_steps 500 \
    --save_total_limit 10 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir $CKPT \
    --dataloader_num_workers 256 \
    --model_author JimmyMa99 \
    --model_name InternLM2.5-1.8B-Lora \
    > "$LOG_FILE" 2>&1 &

# 打印进程ID和日志文件位置
echo "Training started with PID $!"
echo "Log file: $LOG_FILE"

# 显示查看日志的命令
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"
cp $LOG_FILE  ${CKPT}/