import os
import yaml
import torch
import logging
from types import SimpleNamespace
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def get_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("Gemini_Trainer")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def load_config_as_args(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    def dict_to_ns(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
        return ns
    return dict_to_ns(cfg)

def setup_labels(args, logger):
    """建立严格 ID 映射"""
    label_list = getattr(args.dataset, "label_list", [])
    if not isinstance(label_list, list) or len(label_list) == 0:
        msg = "Critical: 'label_list' must be a non-empty list in YAML!"
        logger.error(msg)
        raise ValueError(msg)
    
    label2id = {str(label): i for i, label in enumerate(label_list)}
    id2label = {i: str(label) for i, label in enumerate(label_list)}
    
    logger.info(f"Mapping Table: {label2id}")
    return label_list, label2id, id2label

def get_model(args, id2label, label2id):
    num_labels = len(id2label)
    config_kwargs = {"num_labels": num_labels, "id2label": id2label, "label2id": label2id, "torch_dtype": "auto"}

    if getattr(args.training, "qlora", False):
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, quantization_config=bnb_config, device_map="auto", **config_kwargs
        )
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.training.lora_r, lora_alpha=args.training.lora_alpha,
            lora_dropout=0.1, target_modules=["query", "key", "value"], task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **config_kwargs)
    return model

def print_trainable_parameters(model, logger):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable} / {total} ({100*trainable/total:.2f}%)")

    
if __name__ == "__main__":
    cfg_path = "configs/fp16.yaml"
    args = load_config_as_args(cfg_path)
    print(args)