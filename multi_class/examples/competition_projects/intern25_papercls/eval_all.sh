#!/bin/bash

# --- 配置部分 ---

# 要搜索 checkpoint 的根目录
# 示例: ./swift_output/InternLM2.5-1.8B-Lora-1w3/
# 将会查找类似 './swift_output/InternLM2.5-1.8B-Lora-1w3/checkpoint-1000' 这样的目录
# SEARCH_ROOT_DIR="./swift_output/InternLM2.5-1.8B-Lora-1w3/v0-20250713-182022/" # 根据你的实际情况修改
SEARCH_ROOT_DIR=="./swift_output/InternLM2.5-1.8B-Lora-1w3-multi-template/v2-20250717-012826/"
# 评估脚本的路径
EVAL_SCRIPT="./eval.py"

# 数据集路径
DATASET_PATH="./data/valid_mt_bert.jsonl"

# 批量大小
BATCH_SIZE=2

# CUDA_VISIBLE_DEVICES 设置
CUDA_DEVICES="0"

# --- 脚本逻辑 ---

# 检查必需的命令是否存在
# command -v swift >/dev/null 2>&1 || { echo >&2 "错误: 'swift' 命令未找到。请确保 Swift LLM 已安装并配置在 PATH 中。"; exit 1; }
# command -v python3 >/dev/null 2>&1 || { echo >&2 "错误: 'python3' 命令未找到。请确保 Python 3 已安装。"; exit 1; }
# command -v grep >/dev/null 2>&1 || { echo >&2 "错误: 'grep' 命令未找到。"; exit 1; }
# command -v jq >/dev/null 2>&1 || { echo >&2 "错误: 'jq' 命令未找到。请安装它 (e.g., sudo apt-get install jq)。"; exit 1; }

# 检查必需的输入文件和目录是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "错误: 评估脚本 '$EVAL_SCRIPT' 不存在。请检查 EVAL_SCRIPT 变量。"
  exit 1
fi
if [ ! -f "$DATASET_PATH" ]; then
  echo "错误: 数据集文件 '$DATASET_PATH' 不存在。请检查 DATASET_PATH 变量。"
  exit 1
fi
if [ ! -d "$SEARCH_ROOT_DIR" ]; then
  echo "错误: 搜索根目录 '$SEARCH_ROOT_DIR' 不存在。请检查 SEARCH_ROOT_DIR 变量。"
  exit 1
fi

echo "=================================================="
echo "🚀 开始自动化评估流程"
echo "=================================================="
echo "搜索根目录: $SEARCH_ROOT_DIR"
echo "评估脚本: $EVAL_SCRIPT"
echo "数据集: $DATASET_PATH"
echo "批量大小: $BATCH_SIZE"
echo "CUDA 设备: $CUDA_DEVICES"
echo "=================================================="

# 存储所有 checkpoint 的信息和评估结果
declare -a all_checkpoints_metrics

# 查找所有 checkpoint-* 目录
# 使用 find 命令递归查找，匹配 'checkpoint-数字' 格式
find "$SEARCH_ROOT_DIR" -type d -regex ".*/checkpoint-[0-9]+" | sort | while read -r checkpoint_dir; do
    echo ""
    echo "=================================================="
    echo "正在处理 Checkpoint 目录: $checkpoint_dir"
    echo "=================================================="

    # 确定 merged 目录路径
    MERGED_DIR="${checkpoint_dir}/merged"
    EVAL_OUTPUT_DIR="${checkpoint_dir}/evaluation_results"
    MERGE_LORA_FLAG=true # 假设总是需要合并（除非 merged 已经存在）

    # 检查 merged 目录是否已存在
    if [ -d "$MERGED_DIR" ]; then
        echo "检测到已存在的 Merged 目录: $MERGED_DIR"
        # 如果 merged 目录存在，我们认为合并已经完成，不需要再次执行
        MERGE_LORA_FLAG=false
        # 但我们仍然需要一个模型路径来评估，如果 merged 目录是模型本身（例如直接合并到 checkpoint-xxx 目录），则模型路径就是 checkpoint_dir
        # 如果 merged 是一个子目录，则模型路径是 $MERGED_DIR
        MODEL_TO_EVALUATE="$MERGED_DIR"
    else
        # 如果 merged 目录不存在，则需要执行导出合并
        echo "未检测到 Merged 目录，正在执行 Swift LoRA 合并..."
        # Swift 导出的模型路径是 MERGED_DIR，所以命令中的 --output_dir 就是 MERGED_DIR
        MERGE_COMMAND="swift export --adapters \"$checkpoint_dir\" --merge_lora true --output_dir \"$MERGED_DIR\""
        echo "执行命令: $MERGE_COMMAND"

        # 执行合并命令
        eval $MERGE_COMMAND
        MERGE_EXIT_CODE=$?

        if [ $MERGE_EXIT_CODE -eq 0 ]; then
            echo "✔️ Swift LoRA 合并成功。模型位于: $MERGED_DIR"
            MODEL_TO_EVALUATE="$MERGED_DIR"
        else
            echo "❌ Swift LoRA 合并失败，退出代码: $MERGE_EXIT_CODE。跳过此 checkpoint 的评估。"
            continue # 跳过当前循环的剩余部分，处理下一个 checkpoint
        fi
    fi

    # 检查要评估的模型路径是否存在
    if [ ! -d "$MODEL_TO_EVALUATE" ]; then
        echo "错误: 模型路径 '$MODEL_TO_EVALUATE' 不存在或不是一个目录。跳过此 checkpoint 的评估。"
        continue
    fi

    # 检查是否需要进行评估 (即，如果 merged 存在，我们仍然可以重新评估，但可以根据需要添加条件)
    # 这里我们总是尝试评估，因为即使 merged 存在，也可能想要重新验证
    # 如果想避免重复评估已有的 eval results，可以加如下判断：
    # if [ -f "${EVAL_OUTPUT_DIR}/metric.json" ]; then
    #     echo "检测到已有的评估结果，跳过重复评估..."
    #     # 如果要读取已有的结果，需要在这里加载
    # else
        echo "开始对模型 '$MODEL_TO_EVALUATE' 进行评估..."
        EVAL_COMMAND="python3 $EVAL_SCRIPT --input_dataset \"$DATASET_PATH\" --model_path \"$MODEL_TO_EVALUATE\" --output_dir \"$EVAL_OUTPUT_DIR\" --batch_size $BATCH_SIZE --cuda_visible_devices \"$CUDA_DEVICES\""
        echo "执行命令: $EVAL_COMMAND"

        # 执行评估命令
        eval $EVAL_COMMAND
        EVAL_EXIT_CODE=$?

        if [ $EVAL_EXIT_CODE -eq 0 ]; then
            echo "✔️ 评估成功。结果保存在: $EVAL_OUTPUT_DIR"

            # 提取 accuracy
            METRIC_FILE="${EVAL_OUTPUT_DIR}/metric.json"
            if [ -f "$METRIC_FILE" ]; then
                # 使用 jq 来提取 accuracy 字段
                # 确保 'jq' 命令已安装
                accuracy=$(jq -r '.accuracy' "$METRIC_FILE")
                if [ "$accuracy" == "null" ]; then
                    echo "警告: 在 $METRIC_FILE 中未找到 'accuracy' 字段。"
                else
                    # 将 checkpoint 目录名（通常是最后一个组件，如 'checkpoint-1000'）提取出来
                    checkpoint_name=$(basename "$checkpoint_dir")
                    all_checkpoints_metrics+=("$checkpoint_name:$accuracy")
                    echo "提取到 Accuracy for $checkpoint_name: $accuracy"
                fi
            else
                echo "警告: 未找到评估指标文件: $METRIC_FILE"
            fi
        else
            echo "❌ 评估失败，退出代码: $EVAL_EXIT_CODE。"
        fi
    # fi # 结束上面可选的 if 判断
done

echo ""
echo "=================================================="
echo "🌟 所有 Checkpoint 评估汇总 🌟"
echo "=================================================="

if [ ${#all_checkpoints_metrics[@]} -eq 0 ]; then
    echo "未找到任何成功的评估结果，无法生成汇总报告。"
else
    # 打印汇总报告
    echo "Checkpoint Name | Accuracy"
    echo "----------------|----------"
    # 遍历数组并打印，对齐格式
    for metric_entry in "${all_checkpoints_metrics[@]}"; do
        # 分割 checkpoint_name 和 accuracy
        IFS=':' read -r ckpt_name acc_value <<< "$metric_entry"
        # 打印并格式化对齐
        printf "%-15s | %s\n" "$ckpt_name" "$acc_value"
    done
    echo "=================================================="
fi

echo "🚀 自动化评估流程完成。"
echo "=================================================="

