
# 准备数据

bash data_scripts/download_and_preprocess.sh 

# 准备代码
cp  ../../../*.py .
cp  ../../../scripts .

# 训练
bash experiments/run_all.sh 