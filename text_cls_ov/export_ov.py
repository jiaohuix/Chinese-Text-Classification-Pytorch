import argparse
from optimum.intel import OVModelForSequenceClassification, OVWeightQuantizationConfig
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Export Model to OpenVINO (INT8/INT4)")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="PyTorch 合并后的模型路径")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="OpenVINO 模型保存路径")
    parser.add_argument("-q", "--quant", choices=["fp16", "int8", "int4"], default="int8", help="量化位数")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 根据选择配置量化参数
    if args.quant == "int4":
        # 参考文档：OVWeightQuantizationConfig(bits=4)
        print("[*] 正在执行 INT4 权重量化 (Weight-only)...")
        q_config = OVWeightQuantizationConfig(bits=4, sym=True)
        model = OVModelForSequenceClassification.from_pretrained(
            args.model_path, export=True, quantization_config=q_config
        )
    elif args.quant == "int8":
        print("[*] 正在执行 INT8 量化...")
        # 简单高效的 int8 导出
        model = OVModelForSequenceClassification.from_pretrained(
            args.model_path, export=True, load_in_8bit=True
        )
    else:
        print("[*] 正在导出 FP16 精度模型...")
        model = OVModelForSequenceClassification.from_pretrained(
            args.model_path, export=True
        )

    # 保存结果
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[✔] 已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()