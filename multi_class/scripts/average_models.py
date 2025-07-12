'''
python scripts/average_models.py --models ckpt/patient_intention/v25/v25_doubao_bsz16/checkpoint-1000/model.safetensors  ckpt/patient_intention/v25/v25_doubao_bsz16/checkpoint-1200/model.safetensors ckpt/patient_intention/v25/v25_doubao_bsz16/checkpoint-1600/model.safetensors --output ckpt/patient_intention/v25/v25_doubao_bsz16/averaged_model.safetensors
'''
import os
import argparse
from safetensors.torch import load_file, save_file
import torch
from tqdm import tqdm
import warnings

def validate_model_paths(model_paths):
    """验证所有模型路径是否存在且可读"""
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        if not path.endswith('.safetensors'):
            warnings.warn(f"非标准SafeTensors文件: {path}")

def average_models(model_paths, output_path, weights=None):
    """
    Average multiple SafeTensors models with enhanced safety checks
    
    Args:
        model_paths (list): List of paths to .safetensors files
        output_path (str): Full output path (must end with .safetensors)
        weights (list, optional): List of weights for each model
    """
    # ===== 输入验证 =====
    validate_model_paths(model_paths)
    
    if not output_path.endswith('.safetensors'):
        raise ValueError("输出路径必须以.safetensors结尾")
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # ===== 权重处理 =====
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        if len(weights) != len(model_paths):
            raise ValueError("权重数量必须与模型数量相同")
        weights = [float(w) for w in weights]  # 强制转换为float
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # 归一化
    
    # ===== 模型加载 =====
    try:
        first_model = load_file(model_paths[0])
    except Exception as e:
        raise RuntimeError(f"加载第一个模型失败: {str(e)}")
    
    # ===== 平均计算 =====
    averaged_model = {k: torch.zeros_like(v) for k, v in first_model.items()}
    
    for model_path, weight in tqdm(zip(model_paths, weights), 
                                 total=len(model_paths),
                                 desc="Averaging models"):
        try:
            model = load_file(model_path)
            # 检查模型结构一致性
            if set(model.keys()) != set(averaged_model.keys()):
                raise ValueError(f"模型结构不一致: {model_path}")
            
            for key in model:
                if model[key].shape != averaged_model[key].shape:
                    raise ValueError(f"张量形状不匹配[{key}]: {model[key].shape} vs {averaged_model[key].shape}")
                averaged_model[key] += model[key] * weight
                
        except Exception as e:
            raise RuntimeError(f"处理模型 {model_path} 时出错: {str(e)}")
    
    # ===== 保存结果 =====
    try:
        save_file(averaged_model, output_path)
        print(f"✅ 成功保存平均模型到: {output_path}")
        print(f"▸ 融合模型数: {len(model_paths)}")
        print(f"▸ 使用权重: {weights}")
    except Exception as e:
        raise RuntimeError(f"保存模型失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="安全地平均多个SafeTensors模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--models", nargs="+", required=True,
                      help="需要平均的.safetensors模型路径列表")
    parser.add_argument("--output", required=True,
                      help="输出路径（必须以.safetensors结尾）")
    parser.add_argument("--weights", nargs="+", type=float,
                      help="对应每个模型的权重（可选）")
    
    args = parser.parse_args()
    
    try:
        average_models(args.models, args.output, args.weights)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
