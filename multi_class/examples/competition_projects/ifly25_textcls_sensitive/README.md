
# 准备数据

bash data_scripts/download_and_preprocess.sh 

# 准备代码
cp  ../../../*.py .
cp -r ../../../scripts .

# 训练
python scripts/modelscope_downloader.py
bash experiments/run_all.sh 

# 预测
python predict.py data/processed/test.jsonl data/processed/test.csv  ckpt
