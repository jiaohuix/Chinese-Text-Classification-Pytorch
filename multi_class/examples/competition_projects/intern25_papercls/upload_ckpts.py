'''
魔乐的安装
https://modelers.cn/docs/zh/openmind-hub-client/0.9/install.html

'''
import os
import time
from openmind_hub import upload_folder, create_repo
from pathlib import Path

TOKEN = "f9170d3fdcf9be97d7d0b90da4b9679e3917b7c3"
# REPO_PREFIX = "camp_test_jhx25/cls26-04-1w3-mpromt"
# # MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3/v0-20250713-182022/"
# MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3-multi-template/v2-20250717-012826/"
# # CKPTS = ["checkpoint-2000", "checkpoint-4000", "checkpoint-6000"]
# # CKPTS = ["checkpoint-3000","checkpoint-4000", "checkpoint-5000", "checkpoint-6000"]
# CKPTS = ["checkpoint-4000", "checkpoint-5000", "checkpoint-6000"]



# # 25/07/18
# REPO_PREFIX = "camp_test_jhx25/cls26-05-2w4"
# MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-2w4/v1-20250718-001014/"
# CKPTS = ["checkpoint-2000","checkpoint-4000", "checkpoint-6000", "checkpoint-8000"]

# # day0720 - V类添加
# REPO_PREFIX = "camp_test_jhx25/cls26-05-1w3v21-vclass"
# MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3v21/v1-20250720-191116/"
# CKPTS = ["checkpoint-1000", "checkpoint-1637"]

# # day0721 - V类添加到原始的1w2数据（效果差）
# REPO_PREFIX = "camp_test_jhx25/cls26-05-1w3v22-vclass"
# MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3v22/v0-20250721-100716"
# CKPTS = ["checkpoint-1000", "checkpoint-2000", "checkpoint-3000"]
# # CKPTS = [ "checkpoint-3000"]

# # day0721  多类别采样，多模板 90%原始模板
# REPO_PREFIX = "camp_test_jhx25/cls26-05-1w3v22-mprompt0_9"
# MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3v22-mprompt0.9/v1-20250721-215834/"
# CKPTS = ["checkpoint-1000", "checkpoint-2000", "checkpoint-3000"]



# # day0721  多类别采样，多模板 90%原始模板
# REPO_PREFIX = "camp_test_jhx25/cls26-05-1w3v22-mprompt0_5"
# MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3v22-mprompt0.5/v0-20250722-002724/"
# # CKPTS = ["checkpoint-1000", "checkpoint-2000", "checkpoint-3000"]
# # CKPTS = ["checkpoint-3000"]
# CKPTS = ["checkpoint-4000", "checkpoint-4914"]



# day0721  多类别采样，单模板，neftune alpha1
REPO_PREFIX = "camp_test_jhx25/cls26-05-1w3v-neftune1"
MODEL_ROOT = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora-1w3v21-neft1/v0-20250722-103656/"
# CKPTS = ["checkpoint-1000", "checkpoint-2000", "checkpoint-3000"]
# CKPTS = ["checkpoint-3000"]
CKPTS = ["checkpoint-4911", "checkpoint-4000", "checkpoint-3000"]


def print_log(message, symbol="ℹ️", color=None):
    """打印格式化的日志信息"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m"
    }
    color_code = colors.get(color, "")
    print(f"{color_code}{symbol} {message}{colors['end'] if color else ''}")

def check_directory_exists(path):
    """检查目录是否存在"""
    path = Path(path)
    if not path.exists():
        print_log(f"目录不存在: {path}", "❌", "red")
        return False
    if not path.is_dir():
        print_log(f"路径不是目录: {path}", "❌", "red")
        return False
    return True

def create_and_upload_models(model_dirs):
    """创建模型仓库并上传模型"""
    total_start = time.time()
    print_log(f"开始处理 {len(model_dirs)} 个模型目录", "🚀", "blue")
    
    for i, model_dir in enumerate(model_dirs, 1):
        step_start = time.time()
        step = model_dir.split("-")[-1].replace("/merged","").strip()
        REPO_ID = REPO_PREFIX + f"-{step}"
        
        print_log(f"正在处理第 {i}/{len(model_dirs)} 个模型: {model_dir}", "🔍", "yellow")
        
        # 检查目录是否存在
        if not check_directory_exists(model_dir):
            continue
 
        try:
            # 创建仓库
            print_log(f"创建模型仓库: {REPO_ID}", "🛠️", "blue")
            create_repo(
                token=TOKEN,
                repo_id=REPO_ID,
                repo_type="model"
            )
        except Exception as e:
            print_log(f"创建仓库 {REPO_ID} 失败: {str(e)}", "❌", "red")

        try:
            # 上传模型
            print_log(f"开始上传模型: {model_dir}", "📤", "blue")
            upload_start = time.time()
            
            upload_folder(
                token=TOKEN,
                folder_path=model_dir,
                repo_id=REPO_ID,
            )
            
            upload_time = time.time() - upload_start
            print_log(f"上传完成! 耗时: {upload_time:.2f}s", "✅", "green")
            
        except Exception as e:
            print_log(f"上传模型 {model_dir} 时出错: {str(e)}", "❌", "red")
            continue


        step_time = time.time() - step_start
        print_log(f"模型 {step} 处理完成! 总耗时: {step_time:.2f}s\n", "✔️", "green")
    
    total_time = time.time() - total_start
    print_log(f"所有模型处理完成! 总耗时: {total_time:.2f}s", "🎉", "green")

def main():
    # 检查根目录是否存在
    if not check_directory_exists(MODEL_ROOT):
        return
        
    # 准备模型目录列表
    model_dirs = [os.path.join(MODEL_ROOT, ckpt, "merged") for ckpt in CKPTS]
    
    # 过滤掉不存在的目录
    valid_dirs = [d for d in model_dirs if check_directory_exists(d)]
    
    if not valid_dirs:
        print_log("没有有效的模型目录可处理", "⚠️", "red")
        return
        
    print_log(f"找到 {len(valid_dirs)} 个有效模型目录", "📂", "green")
    
    # 开始处理
    create_and_upload_models(valid_dirs)

if __name__ == "__main__":
    main()
