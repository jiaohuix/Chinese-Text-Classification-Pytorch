'''
@date: 2025-07-12 13:09
@author: jiaohuix
@description:
    该脚本用于从ModelScope下载预训练模型。
    它主要用于下载特定模型（例如chinese-roberta-wwm-ext），
    并可以配置下载的缓存目录、最大工作线程数以及忽略特定文件类型。

    使用方法:
    python download_model.py

    在执行脚本前，请确保已安装modelscope库（pip install modelscope）。
'''
#
import os
import logging
from modelscope.hub.snapshot_download import snapshot_download

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_pretrained_model(model_id: str, cache_path: str, revision: str = 'master', max_workers: int = 8, ignore_patterns: list = None):
    """
    从ModelScope下载预训练模型。

    Args:
        model_id (str): 要下载的模型ID (例如: 'dienstag/chinese-roberta-wwm-ext')。
        cache_path (str): 模型下载的本地缓存目录。
        revision (str): 模型版本，默认为 'master'。
        max_workers (int): 下载时使用的最大工作线程数，默认为8。
        ignore_patterns (list): 要忽略下载的文件模式列表。
    """
    logging.info(f"🚀 正在从ModelScope下载模型: '{model_id}'...")

    try:
        model_dir = snapshot_download(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_path,
            max_workers=max_workers,
            ignore_patterns=ignore_patterns
        )
        logging.info(f"✅ 模型下载成功！模型已保存在: {model_dir}")
        return model_dir
    except Exception as e:
        logging.error(f"❌ 模型下载失败: {e}")
        return None

if __name__ == "__main__":
    # --- 用户输入 ---
    # 指定要下载的模型ID
    MODEL_ID = 'dienstag/chinese-roberta-wwm-ext'
    # 指定模型下载的本地缓存目录
    CACHE_DIR = './models'
    # 指定模型版本，通常使用 'master' 或特定的git commit hash
    REVISION = 'master'
    # 设置下载时使用的最大工作线程数
    MAX_WORKERS = 8
    # 指定需要忽略的文件类型，以避免下载不必要的文件
    # 例如，如果你只需要PyTorch模型，可以忽略TensorFlow和Flax格式的模型文件
    IGNORE_PATTERNS = ['tf_model.h5', 'flax_model.msgpack']
    # --- 用户输入结束 ---

    downloaded_model_path = download_pretrained_model(
        model_id=MODEL_ID,
        cache_path=CACHE_DIR,
        revision=REVISION,
        max_workers=MAX_WORKERS,
        ignore_patterns=IGNORE_PATTERNS
    )

    if downloaded_model_path:
        print(f"\n模型 '{MODEL_ID}' 的下载路径为: {downloaded_model_path}")
