论文分类比赛

# 环境

pip install uv
uv venv ms-swift -p python3.10
source ms-swift/bin/activate
uv pip install 'ms-swift' -i  https://pypi.tuna.tsinghua.edu.cn/simple

# 数据处理

下载模型
临时取消所有代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

modelscope download --repo-type model Shanghai_AI_Laboratory/internlm2_5-1_8b-chat --local_dir internlm2_5-1_8b-chat
下载数据：
modelscope download --dataset livehouse/arXiv-metadata-oai-snapshot-ver233  --local_dir  data_arxiv

python process.py
bash train_test_split.sh
输入需要data/corpus.jsonl，输出data/train.jsonl data/valid.jsonl(2k5)

```
python data_scripts/process_v4_multi_class_mprompt.py
```
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


python data_scripts/check_length.py --model_path /root/train-paper
/code/Chinese-Text-Classification-Pytorch-master/multi_class/examples/competition_projects/ifly25_textcls_sensitive/models/dienstag/chinese-roberta-wwm-ext/ --data_path data/train_bert.jsonl 

## 多模板

python data_scripts/process_multi_template.py
bash data_scripts/train_test_split.sh
python data_scripts/convert_chat_to_bert.py -i data/train_mt.jsonl -o data/train_mt_bert.jsonl 
python data_scripts/convert_chat_to_bert.py -i data/valid_mt.jsonl -o data/valid_mt_bert.jsonl 
```
Generating train split: 2500 examples [00:00, 22018.45 examples/s]
✅ 成功加载数据，共 2500 条样本
📊 数据结构: Dataset({
    features: ['system', 'conversation'],
    num_rows: 2500
})
🏷️ 列名: ['system', 'conversation']
🔄 开始转换数据格式...
Processing: 100%|██████████████████████████████████████████████████████████████████| 2500/2500 [00:00<00:00, 8972.28 examples/s]
✅ 数据转换完成
💾 正在保存到 data/valid_mt_bert.jsonl...
Creating json from Arrow format: 100%|████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 38.11ba/s]
✅ 成功保存 2500 条样本
📄 输出文件: /root/train-paper/code/data/valid_mt_bert.jsonl

🔍 转换后样例:
{
  "text": "LIBRARY CLASSIFICATION CARD\nNotes: Comments: American Control Conference (ACC) 2024 DOI: None\nTitle: Multimodal Safe Control for Human-Robot Interaction\nAuthor(s): Ravi Pandya, Tianhao Wei, Changliu Liu\nAbstract:   Generating safe behaviors for autonomous systems is important as they\ncontinue to be deployed in the real world
```