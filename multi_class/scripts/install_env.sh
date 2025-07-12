#!/bin/bash
VENV_NAME=".bert_cls"

# 1. 安装 uv（如果尚未安装）
if ! command -v uv >/dev/null 2>&1; then
    echo "🚀 正在安装 uv..."
    pip install uv
    echo "✅ uv安装完成。"
fi

# 2. 创建虚拟环境（如果不存在）
if [ ! -d "$VENV_NAME" ]; then
    echo "🚀 正在创建虚拟环境 $VENV_NAME..."
    uv venv $VENV_NAME -p python3.10
else
    echo "✅ 虚拟环境 $VENV_NAME 已存在，跳过创建。"
fi

# 3. 激活虚拟环境并安装依赖
echo "🚀 正在安装依赖 $VENV_NAME..."
source $VENV_NAME/bin/activate && \
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 提示用户
echo "✅ 环境已就绪！当前 Python 路径：$(which python)"
echo "如需手动激活虚拟环境，运行："
echo "source $VENV_NAME/bin/activate"
