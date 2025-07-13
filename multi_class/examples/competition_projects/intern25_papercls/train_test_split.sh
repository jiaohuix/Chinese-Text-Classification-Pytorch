data="data/sample_data_600.jsonl"
# 1. 随机打乱原始文件并分割
shuf $data -o ${data}_tmp

# 2. 提取前2500行作为验证集
head -n 2500 ${data}_tmp > data/valid.jsonl

# 3. 剩余部分作为训练集
tail -n +2501 ${data}_tmp > data/train.jsonl

# 4. 删除临时文件（可选）
rm  ${data}_tmp
