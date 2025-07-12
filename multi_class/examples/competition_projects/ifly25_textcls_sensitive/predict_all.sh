fold=1
python predict.py data/processed/test.jsonl data/processed/predict_fold${fold}.csv ckpt/ifly25_baseline_fold${fold} data/processed/labels.json
fold=2
python predict.py data/processed/test.jsonl data/processed/predict_fold${fold}.csv ckpt/ifly25_baseline_fold${fold} data/processed/labels.json
fold=3
python predict.py data/processed/test.jsonl data/processed/predict_fold${fold}.csv ckpt/ifly25_baseline_fold${fold} data/processed/labels.json
fold=4
python predict.py data/processed/test.jsonl data/processed/predict_fold${fold}.csv ckpt/ifly25_baseline_fold${fold} data/processed/labels.json
fold=5
python predict.py data/processed/test.jsonl data/processed/predict_fold${fold}.csv ckpt/ifly25_baseline_fold${fold} data/processed/labels.json