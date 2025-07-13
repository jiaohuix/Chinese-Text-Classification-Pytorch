论文分类比赛

# 环境

pip install uv
uv venv ms-swift -p python3.10
source ms-swift/bin/activate
uv pip install 'ms-swift' -i  https://pypi.tuna.tsinghua.edu.cn/simple

# 数据处理
python process.py
bash train_test_split.sh
输入需要data/corpus.jsonl，输出data/train.jsonl data/valid.jsonl(2k5)

python data_scripts/convert_chat_to_bert.py -i data/train.jsonl -o data/train_bert.jsonl
python data_scripts/convert_chat_to_bert.py -i data/valid.jsonl -o data/valid_bert.jsonl

# 训练

bash paper_sft_ms-swift.sh


# 合并权重

time swift export --adapters swift_output/InternLM2.5-1.8B-Lora/v2-20250711-000411/checkpoint-1000/ --merge_lora true --output_dir /root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora/merged/

# 上传模型

python upload.py


# 旧版的数据处理

https://colab.research.google.com/drive/1E-BFn_y_T24WC6X3Ab3xvQQ2Bpaeykdp?usp=sharing
!modelscope download --dataset JimmyMa99/smartflow-arxiv-dataset --local_dir ./data

livehouse/arXiv-metadata-oai-snapshot-ver233

modelscope download --dataset livehouse/arXiv-metadata-oai-snapshot-ver233 --local_dir ./data_arxiv
