文本分类


项目结构：
```
multi_class
    train.py 
    /scripts
    /experiments
    /examples
        /research_projects
            llm_cls_swift
            llm_cls_unsloth
            distilation
            quantization
        /competition_projects
            ifly25_textcls_sensitive
            intern25_textcls_paper26
    /data_augmentation
    /deploy
    /analysis
```

环境：
```
pip install uv 
uv venv bert_cls -p python3.10
source bert_cls/bin/activate
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
cd source && unzip metric.zip && cp -r metric/accuracy ..
```
或者
```
 bash scripts/install_env.sh 
```

新闻分类例子：

数据处理：
cd multi_class
python scripts/thunews_preprocess.py 

下载模型
python scripts/modelscope_downloader.py 

训练
bash experiments/run_all.sh 

python inference.py -i data/test.jsonl -o results -m ckpt -l ckpt/labels.json --do_eval --batch_size 32


| 模型类型       | 样本数量 | 准确率  | 错误案例数 | 总执行时间（秒） | QPS   |
|----------------|----------|---------|------------|------------------|-------|
| FP32（原模型） | 2002     | 0.8531  | 294        | 14.5618          | 137.48|
| FP32（ONNX）   | 2002     | 0.8531  | 294        | 13.8005          | 145.07|
| INT8（ONNX量化）| 2002     | 0.7622  | 476        | 10.7145          | 186.85|
