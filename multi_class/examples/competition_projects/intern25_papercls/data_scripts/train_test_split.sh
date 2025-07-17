# data="data/sample_data_600.jsonl"
# valid_vile=data/valid.jsonl
# train_file=data/train.jsonl

data="data/sample_data_600_multi_template.jsonl"
valid_vile=data/valid_mt.jsonl
train_file=data/train_mt.jsonl


# 1. 随机打乱原始文件并分割
shuf $data -o ${data}_tmp

# 2. 提取前2500行作为验证集
head -n 2500 ${data}_tmp > $valid_vile

# 3. 剩余部分作为训练集
tail -n +2501 ${data}_tmp > $train_file

# 4. 删除临时文件（可选）
rm  ${data}_tmp
