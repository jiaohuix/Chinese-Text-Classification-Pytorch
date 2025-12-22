import argparse
import random
import numpy as np
import torch
import evaluate
import os
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer, set_seed as hf_set_seed
from datasets import load_dataset
from utils import load_config_as_args, get_model, setup_labels, get_logger, print_trainable_parameters

def apply_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = load_config_as_args(parser.parse_args().config)
    
    logger = get_logger(args.training.output_dir)
    seed = getattr(args.training, "seed", 42)
    apply_seed(seed)

    # 1. 标签初始化
    label_list, label2id, id2label = setup_labels(args, logger)
    
    # 2. 数据加载
    files = {"train": args.dataset.train_path, "validation": args.dataset.val_path}
    raw_datasets = load_dataset("json", data_files=files)
    train_ds = raw_datasets["train"].shuffle(seed=getattr(args.dataset, "seed", seed))
    val_ds = raw_datasets["validation"].shuffle(seed=getattr(args.dataset, "seed", seed))

    # 3. 预处理逻辑与三因素条件校验
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 判断是否满足文本转换的三要素
    use_text_flag = getattr(args.dataset, "use_text_label_flag", False)
    label_text_col = getattr(args.dataset, "label_text_column", None)
    
    # 条件检查：Flag开启 AND 列名存在于数据中 AND label_list不为空(setup_labels已检查)
    can_convert_text = use_text_flag and (label_text_col in train_ds.column_names)
    
    source_col = label_text_col if can_convert_text else args.dataset.label_column
    logger.info(f"Label Source: {'[TEXT] ' + source_col if can_convert_text else '[ID] ' + source_col}")

    def preprocess_fn(batch):
        result = tokenizer(batch[args.dataset.text_column], truncation=True, max_length=128)
        
        def safe_map(val):
            s_val = str(val)
            if s_val not in label2id:
                # 校验失败：打印详细错误并中断
                err = f"Label Mapping Error! Value '{s_val}' not in label_list. \nContext: {val}"
                logger.error(err)
                raise ValueError(err)
            return label2id[s_val]

        # 满足条件则执行文本->ID转换，否则假设输入已是ID并做校验
        result["labels"] = list(map(safe_map, batch[source_col]))
        return result

    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=val_ds.column_names)

    # 4. 训练
    model = get_model(args, id2label, label2id)
    print_trainable_parameters(model, logger)

    training_args = TrainingArguments(
        output_dir=args.training.output_dir,
        learning_rate=float(args.training.lr),
        num_train_epochs=args.training.epochs,
        per_device_train_batch_size=args.training.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.training.logging_steps,
        report_to="tensorboard",
        load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported()
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {"accuracy": evaluate.load("accuracy").compute(predictions=p.predictions.argmax(-1), references=p.label_ids)["accuracy"]}
    )

    trainer.train()
    
    if getattr(args.training, "qlora", False):
        model.save_pretrained(os.path.join(args.training.output_dir, "best_adapter"))

if __name__ == "__main__":
    main()