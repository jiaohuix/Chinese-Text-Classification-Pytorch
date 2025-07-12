#!/bin/bash
# data_scripts/download_competition_data.sh

# 设置颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 修正后的日志函数
log_info() {
    echo -e "${BLUE}ℹ️  $*${NC}"
}

log_success() {
    echo -e "${GREEN}✅  $*${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

log_error() {
    echo -e "${RED}❌  $*${NC}"
}

DATA_DIR="data/"
mkdir -p ${DATA_DIR}

log_info "🚀 开始下载比赛数据集..."

# 下载主数据集（含重试机制）
MAX_RETRY=3
for ((i=1; i<=${MAX_RETRY}; i++))
do
    log_info "📥 尝试第 $i 次下载 (共 $MAX_RETRY 次)..."
    
    wget -c https://bjcdn.openstorage.cn/aicontest/2025%E7%AE%97%E6%B3%95%E8%B5%9B/%E5%9F%BA%E4%BA%8E%E6%96%87%E6%9C%AC%E7%9A%84%E8%BF%9D%E7%A6%81%E8%AF%8D%E5%88%86%E7%B1%BB%E6%8C%91%E6%88%98%E8%B5%9B/dataset.zip \
        -O ${DATA_DIR}/dataset.zip && \
    wget -c https://bjcdn.openstorage.cn/aicontest/2025%E7%AE%97%E6%B3%95%E8%B5%9B/%E5%9F%BA%E4%BA%8E%E6%96%87%E6%9C%AC%E7%9A%84%E8%BF%9D%E7%A6%81%E8%AF%8D%E5%88%86%E7%B1%BB%E6%8C%91%E6%88%98%E8%B5%9B/example.csv \
        -O ${DATA_DIR}/example.csv
    
    if [ $? -eq 0 ]; then
        log_success "文件下载成功！"
        break
    else
        log_warning "下载尝试 $i 失败，5秒后重试..."
        sleep 5
    fi
done

# 校验文件完整性
if [ -f "${DATA_DIR}/dataset.zip" ] && [ -f "${DATA_DIR}/example.csv" ]; then
    log_info "🔍 正在验证文件完整性..."
    
    DS_SIZE=$(stat -c%s "${DATA_DIR}/dataset.zip")
    EX_SIZE=$(stat -c%s "${DATA_DIR}/example.csv")
    
    if [ ${DS_SIZE} -lt 1000000 ] || [ ${EX_SIZE} -lt 100 ]; then
        log_error "下载的文件太小，可能已损坏"
        exit 1
    fi
    
    log_success "文件验证通过！"
    
    # 解压数据集
    log_info "📦 正在解压 dataset.zip..."
    unzip -o ${DATA_DIR}/dataset.zip -d ${DATA_DIR}/
    if [ $? -eq 0 ]; then
        log_success "解压完成！"
    else
        log_error "解压失败"
        exit 1
    fi

    # 转csv为jsonl
    log_info "🔄 正在转换 CSV 为 JSONL 格式..."
    python data_scripts/csv2jsonl.py -i data/dataset/train_all.csv -o data/processed/train.jsonl -t "文本" -l "类别" && \
    python data_scripts/csv2jsonl.py -i data/dataset/test_text.csv -o data/processed/test.jsonl -t "文本" -l "类别" --test --label_map data/processed/labels.json
    
    if [ $? -eq 0 ]; then
        log_success "格式转换完成！"
    else
        log_error "格式转换失败"
        exit 1
    fi

    # 交叉验证
    log_info "✂️ 正在进行交叉验证分割 (5折)..."
    python data_scripts/split_kfolds.py -i data/processed/train.jsonl -o data/kfolds -n 5 --seed 42
    if [ $? -eq 0 ]; then
        log_success "交叉验证分割完成！"
    else
        log_error "交叉验证分割失败"
        exit 1
    fi
    
    # 拷贝测试集
    log_info "📋 正在复制测试集到各折目录..."
    for dir in data/kfolds/fold_*/; do 
        cp data/processed/test.jsonl "$dir" && \
        log_info "  已复制到 ${dir}"
    done
    log_success "测试集复制完成！"

    log_success "🎉 所有数据处理步骤完成！"
else
    log_error "所需文件未成功下载"
    exit 1
fi
