export  CUDA_VISIBLE_DEVICES=0
model_path=/home/wonders/zjh/Projects/pretrained_models/chinese-roberta-wwm-ext/
train_file=data/patient_v25/train.jsonl
valid_file=data/patient_v25/dev2k.jsonl

output_dir=./results/ckpt_distil_roberta_alpha0.8
distill_alpha=0.8
# output_dir=./results/ckpt_distil_random
eval_steps=200
cache_dir="$output_dir/logits_cache"
python train_distil.py \
    --model_name_or_path $model_path \
    --train_file $train_file \
    --validation_file $valid_file \
    --shuffle_train_dataset \
    --text_column_names text \
    --label_column_name label \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1\
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir $output_dir  --overwrite_output_dir \
    --eval_strategy "steps" --eval_steps $eval_steps --save_strategy  "steps" --save_steps  $eval_steps \
    --distill_enabled True \
    --distill_alpha $distill_alpha \
    --distill_temperature 2.0 \
    --teacher_api_url http://localhost:9000/v1/knowledge/logits \
    --use_logits_cache True \
    --logits_cache_dir $cache_dir \
    --num_classes 7  --text_column_names text --fp16 --metric_name accuracy
