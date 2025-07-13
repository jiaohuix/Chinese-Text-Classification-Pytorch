#!/bin/bash

# Default values
# dataset="thunews"
# tag="roberta"
LOG_FILE="logs/run.log"
train_file="data/thunews/train.jsonl"
valid_file="data/thunews/dev.jsonl"
test_file="data/thunews/test.jsonl"
model_path="./models/chinese-roberta-wwm-ext/"
num_train_epochs=5
learning_rate=2e-5
batch_size=16
grad_accum=1
output_dir="ckpt/baseline"
max_seq_length=128
rdrop_alpha=0
eval_steps=500
fgm_eps=1
fgm_enabled=0  # 默认禁用FGM
neftune_enabled=0  # 默认禁用NEFTune
neftune_alpha=5
metric_name=accuracy 

# Help function
print_help() {
  echo "Usage: \$0 [options]"
  echo ""
  echo "Options:"
  echo "  --model_path <path>       Path to the pre-trained model (default: $model_path)"
  echo "  --train_file <path>       Path to the training data file (default: $train_file)"
  echo "  --valid_file <path>       Path to the validation data file (default: $valid_file)"
  echo "  --num_train_epochs <int>  Number of training epochs (default: $num_train_epochs)"
  echo "  --learning_rate <float>   Learning rate (default: $learning_rate)"
  echo "  --batch_size <int> Batch size per device (default: $batch_size)"
  echo "  --grad_accum <int> Gradient accumulation steps (default: $grad_accum)"
  echo "  --output_dir <path>       Output directory for checkpoints (default: $output_dir)"
  echo "  --max_seq_length <int>  Maximum sequence length (default: $max_seq_length)"
  echo "  --rdrop_alpha <float>  Rdrop alpha (default: $rdrop_alpha)"
  echo "  --fgm_esp <float>  FGM EPS (default: $fgm_eps)"
  echo "  --neftune_enabled <bool>  NEFTune enabled (default: $neftune_enabled)"
  echo "  --neftune_alpha <float>  NEFTune alpha (default: $neftune_alpha)"
  echo "  --eval_steps <int>  Evaluation steps (default: $eval_steps)"
  echo "  --metric_name <str>  Evaluation metric name(accuracy/f1) (default: $metric_name)"
  echo "  --help                    Show this help message"
  echo ""
  echo "Example:"
  echo "  \$0 --model_path /path/to/model --train_file /path/to/train.csv --num_train_epochs 10"
  exit 1
}


# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in 
    --model_path)
      model_path="$2" 
      shift 2
      ;;
    --train_file)
      train_file="$2" 
      shift 2
      ;;
    --valid_file)
      valid_file="$2" 
      shift 2
      ;;
    --test_file)
      test_file="$2" 
      shift 2
      ;;
    --num_train_epochs)
      num_train_epochs="$2" 
      shift 2
      ;;
    --learning_rate)
      learning_rate="$2" 
      shift 2
      ;;
    --batch_size)
      batch_size="$2" 
      shift 2
      ;;
    --grad_accum)
      grad_accum="$2" 
      shift 2
      ;;
    --output_dir)
      output_dir="$2" 
      shift 2
      ;;
    --max_seq_length)
      max_seq_length="$2" 
      shift 2
      ;;
    --rdrop_alpha)
      rdrop_alpha="$2" 
      shift 2
      ;;
    --eval_steps)
      eval_steps="$2" 
      shift 2
      ;;
    --metric_name)
      metric_name="$2"  
      shift 2
      ;;
    --fgm_eps)
      fgm_eps="$2"
      shift 2
      ;;
  --fgm_enabled)
      fgm_enabled=1
      shift
      ;;
    --neftune_enabled)
      neftune_enabled=1
      shift
      ;;
    --neftune_alpha)
      neftune_alpha="$2"
      shift 2
      ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" 
      print_help
      exit 1
      ;;
  esac
done



# Check for minimum required parameters
if [[ -z "$train_file" || -z "$valid_file" ]]; then
  echo "Error: --train_file and --valid_file are required."
  print_help
  exit 1
fi

# Training script
if [ $fgm_enabled -eq 1 ]; then
  fgm_args="--fgm_enabled --fgm_epsilon $fgm_eps"
else
  fgm_args=""  # 禁用时为空字符串
fi

if [ $neftune_enabled -eq 1 ]; then
  neftune_args="--neftune_enabled --neftune_alpha $neftune_alpha"
else
  neftune_args=""  # 禁用时为空字符串
fi


echo "🚀 Starting training with the following parameters:"
echo "  Model path: $model_path"
echo "  Train file: $train_file"
echo "  Valid file: $valid_file"
echo "  Test file: $test_file"
echo "  Num epochs: $num_train_epochs"
echo "  Learning rate: $learning_rate"
echo "  Batch size: $batch_size"
echo "  Gradient accumulation steps: $grad_accum"
echo "  Output dir: $output_dir"
echo "  Max Seq Length: $max_seq_length"
echo "  Rdrop alpha: $rdrop_alpha"
echo "  FGM Enabled: $fgm_args"
echo "  FGM EPS: $fgm_eps"
echo "  NEFTune Enabled: $neftune_enabled"
echo "  NEFTune Alpha: $neftune_alpha"
echo "  Evaluation steps: $eval_steps"
echo "  Metric name: $metric_name"

# 创建日志目录
LOG_DIR=$output_dir/logs
mkdir -p $LOG_DIR


# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/roberta_${TIMESTAMP}.log"


# 设置CUDA设备
export NPROC_PER_NODE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

echo "Log file: $LOG_FILE"

# 显示查看日志的命令
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"

# nohup python train.py \
python train.py \
    --model_name_or_path "$model_path" \
    --train_file "$train_file" \
    --validation_file "$valid_file" \
    --shuffle_train_dataset \
    --text_column_names text \
    --label_column_name label \
    --text_column_delimiter "\t" \
    --do_train \
    --do_eval \
    --max_seq_length "$max_seq_length" \
    --per_device_train_batch_size "$batch_size" --gradient_accumulation_steps "$grad_accum" \
    --learning_rate "$learning_rate" \
    --num_train_epochs "$num_train_epochs" \
    --output_dir "$output_dir" \
    --overwrite_output_dir  \
    --eval_strategy "steps" --eval_steps $eval_steps --save_strategy  "steps" --save_steps  $eval_steps \
    --fp16 --rdrop_alpha $rdrop_alpha --plot_loss  --metric_name $metric_name \
    $fgm_args $neftune_args  \
    > "$LOG_FILE" 2>&1 
  
# > "$LOG_FILE" 2>&1 &




# 打印进程ID和日志文件位置
echo "✅ Training started with PID $!"
echo "Log file: $LOG_FILE"

# 显示查看日志的命令
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"


# Evaluation (optional)
ckpt="$output_dir"  # Assuming the last checkpoint is the best
if [ -d "$ckpt" ]; then
  cp $model_path/tokenizer*  "$ckpt"
  eval_dir="$ckpt/eval"
  MODEL_NAME=$(basename "$ckpt")
  echo "--------------------------------------------------" >> "$LOG_FILE"
  echo "Evaluating model: $ckpt" >>  "$LOG_FILE"
  echo "--------------------------------------------------" >>  "$LOG_FILE"
  python eval.py "$test_file" "$ckpt"  "$eval_dir"  >> "$LOG_FILE" 2>&1
  echo "" >> "$LOG_FILE"
else
  echo "Warning: Checkpoint directory '$ckpt' not found. Skipping evaluation."
fi
