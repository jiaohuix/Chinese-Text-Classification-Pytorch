#!/bin/bash

# 训练/验证集分割函数（修复种子文件为空问题）
split_train_valid() {
    local input_file="$1"       # 输入文件路径
    local train_file="$2"       # 训练集输出路径
    local valid_file="$3"       # 验证集输出路径
    local valid_size="$4"       # 验证集行数
    local seed="$5"             # 随机种子

    # -------------- 参数验证 --------------
    # 检查参数完整性
    if [ $# -ne 5 ]; then
        echo "错误：参数数量错误！"
        echo "用法：split_train_valid <输入文件> <训练集> <验证集> <验证集大小> <种子>"
        return 1
    fi

    # 检查输入文件是否存在
    if [ ! -f "$input_file" ]; then
        echo "错误：输入文件不存在 - $input_file"
        return 1
    fi

    # 检查输入文件是否有内容
    local total_lines=$(wc -l < "$input_file")
    if [ "$total_lines" -eq 0 ]; then
        echo "错误：输入文件为空 - $input_file"
        return 1
    fi

    # 检查验证集大小是否合理
    if [ "$valid_size" -le 0 ] || [ "$valid_size" -ge "$total_lines" ]; then
        echo "错误：验证集大小必须为 0 < 数值 < $total_lines"
        return 1
    fi

    # -------------- 核心逻辑：用awk替代shuf（兼容性更好） --------------
    local tmp_file="${input_file}.split.tmp"

    echo "正在分割文件: $input_file"
    echo "总行数: $total_lines"
    echo "训练集行数: $((total_lines - valid_size))"
    echo "验证集行数: $valid_size"
    echo "使用随机种子: $seed"

    # 使用awk实现带种子的随机打乱（避免shuf的--random-source兼容性问题）
    # 原理：用种子初始化随机数，给每行生成随机数后排序，再去除随机数列
    awk -v seed="$seed" '
    BEGIN { srand(seed) }  # 用指定种子初始化随机数生成器
    { print rand(), $0 }  # 每行前加一个随机数
    ' "$input_file" | sort -k1,1n | cut -d' ' -f2- > "$tmp_file"

    # 检查临时文件是否生成成功
    if [ ! -f "$tmp_file" ] || [ $(wc -l < "$tmp_file") -ne "$total_lines" ]; then
        echo "错误：临时文件生成失败 - $tmp_file"
        rm -f "$tmp_file"
        return 1
    fi

    # 分割为验证集和训练集
    head -n "$valid_size" "$tmp_file" > "$valid_file"
    tail -n +$((valid_size + 1)) "$tmp_file" > "$train_file"

    # 清理临时文件
    rm -f "$tmp_file"

    # 输出结果验证
    echo "分割结果："
    echo "验证集行数: $(wc -l < "$valid_file")"
    echo "训练集行数: $(wc -l < "$train_file")"
    echo "完成！"
}

# 调用示例
data="data/mclass_mprompt/ratio0.1/sample_data_600.jsonl"
valid_file="data/mclass_mprompt/ratio0.1/valid00.jsonl"
train_file="data/mclass_mprompt/ratio0.1/train00.jsonl"
valid_size=2500
seed=42

# 必须加引号！防止路径含空格/特殊字符
split_train_valid "$data" "$train_file" "$valid_file" "$valid_size" "$seed"
