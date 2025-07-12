'''
文件: balanced_data_augmentor.py
作者: jiahui
日期: 2025年06月05日
功能描述:
  此脚本用于对中文文本数据进行数据增强和类别均衡，以优化BERT等NLP模型的训练。
  它支持多种增强策略（随机实体替换、同义词替换、近义字替换、随机字删除、随机邻近字置换、等价字替换）。
  核心功能包括：
  1. 自动分析并均衡数据集中各类别样本数量，使其达到最大类别的数量（或用户指定数量）。
  2. 灵活混合多种增强策略，避免单一策略过度增强。
  3. 对增强后的数据进行去重。
  4. 提供详细的日志记录，追踪增强过程。

环境安装:
  1. 确保已安装 Python 3.6 或更高版本。
  2. 安装 nlpcda 库：
     `pip install nlpcda`
  3. （可选）如果你的数据中包含 SimBERT 增强，需手动安装相关依赖：
     `pip install keras==2.3.1 bert4keras==0.7.7 tensorflow-gpu==1.13.1` (或 CPU 版本 `tensorflow==1.13.1`)
     注意：本脚本默认不包含 SimBERT 增强。

如何调用:
  1. 准备你的训练数据：将训练数据保存为 `.jsonl` 格式的文件，每行一个JSON对象，
     包含 'text' (待增强文本), 'text_label' (文本标签) 和 'label' (标签ID)。
     例如：`{"text": "你觉得网购和实体店哪个好？", "text_label": "其他", "label": 6}`
     默认输入文件路径为 `data/train.jsonl`。
  2. （可选）如果你想使用自定义的实体或同义词词典，可以在 `BalancedDataAugmentor` 初始化时通过 `base_file` 参数指定。
  3. 运行脚本：
     `python balanced_data_augmentor.py`
  4. 脚本将自动创建 `data` 目录（如果不存在），并在其中生成示例 `train.jsonl` 文件（如果不存在）。
  5. 增强后的数据将保存到 `data/train_aug_new.json` 文件中。
  6. 运行时会打印原始和增强后的类别分布，以及详细的增强日志（如果日志级别设置为 DEBUG）。

数据增强参考项目：
https://github.com/425776024/nlpcda
https://github.com/makcedward/nlpaug
https://paddlenlp.readthedocs.io/zh/stable/dataaug.html
'''
import os
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from nlpcda import Randomword, Similarword, Homophone, RandomDeleteChar, CharPositionExchange, EquivalentChar

class BalancedDataAugmentor:
    def __init__(self, input_path, output_path, max_per_class=None, seed=42):
        self.input_path = input_path
        self.output_path = output_path
        self.max_per_class = max_per_class
        self.seed = seed
        random.seed(seed)
        self.data = []
        self.label2data = defaultdict(list)
        
        # 增强策略及参数优化
        self.aug_strategies = [
            ('randomword', Randomword(create_num=3, change_rate=0.5, seed=seed)), # create_num改为3
            ('similarword', Similarword(create_num=3, change_rate=0.5, seed=seed)), # create_num改为3
            ('homophone', Homophone(create_num=3, change_rate=0.3, seed=seed)),  # create_num改为3
            ('randomdeletechar', RandomDeleteChar(create_num=3, change_rate=0.3, seed=seed)), # create_num改为3
            ('charpositionexchange', CharPositionExchange(create_num=3, change_rate=0.5, char_gram=3, seed=seed)), # create_num改为3
            ('equivalentchar', EquivalentChar(create_num=3, change_rate=0.5, seed=seed)), # create_num改为3
            # baidu_translate / googletrans 需要额外配置或网络，此处暂不包含在自动增强策略中
        ]
        logging.info(f"初始化数据增强器，随机种子: {self.seed}")

    def load_data(self):
        logging.info(f"正在从 {self.input_path} 加载数据...")
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.data.append(item)
                    self.label2data[item['text_label']].append(item)
            logging.info(f"数据加载完成，共 {len(self.data)} 条数据。")
        except FileNotFoundError:
            logging.error(f"错误：文件 {self.input_path} 未找到。请确保文件存在。")
            exit()
        except json.JSONDecodeError as e:
            logging.error(f"错误：文件 {self.input_path} JSON 解析失败: {e}。请检查文件格式。")
            exit()

    def analyze_distribution(self, data=None):
        if data is None:
            data = self.data
        counter = Counter([x['text_label'] for x in data])
        return counter

    def print_distribution(self, counter, title):
        print(f'\n{title}')
        for label, count in sorted(counter.items(), key=lambda x: -x[1]):
            print(f'  {label}: {count}')

    def augment_class(self, label, samples, target_num):
        """
        对某一类别做增强，混合多种策略，循环多轮增强，直到补齐到目标数量。
        """
        ori_num = len(samples)
        if ori_num >= target_num:
            logging.info(f"类别 '{label}' 样本数 ({ori_num}) 已达到或超过目标数 ({target_num})，无需增强。")
            return samples
        
        augmented = []
        strategies = self.aug_strategies
        strategy_idx = 0
        
        logging.info(f"开始增强类别 '{label}'，原始 {ori_num} 条，目标 {target_num} 条。")
        
        while ori_num + len(augmented) < target_num:
            if not samples: # 避免空列表导致random.choice报错
                logging.warning(f"类别 '{label}' 没有可供增强的原始样本，跳过增强。")
                break

            name, strategy = strategies[strategy_idx % len(strategies)]
            sample = random.choice(samples)
            text = sample['text']
            
            try:
                aug_texts = strategy.replace(text) # nlpcda的replace方法通常返回一个列表
                aug_text_candidates = [at for at in aug_texts if at and at != text] # 过滤掉空和与原始相同的
                if aug_text_candidates: # 如果有有效的增强文本
                    aug_text = random.choice(aug_text_candidates) # 从有效增强文本中随机选择一个
                else:
                    aug_text = None # 没有生成有效的新文本

            except Exception as e:
                logging.error(f"增强策略 '{name}' 处理文本 '{text}' 时出错: {e}")
                aug_text = None

            if aug_text: # 确保增强有效
                new_sample = deepcopy(sample)
                new_sample['text'] = aug_text
                augmented.append(new_sample)
                logging.debug(f"  增强日志 - 类别: '{label}', 原始: '{text}', 增强后: '{aug_text}' (策略: {name})") # 打印日志
            else:
                # 如果增强未成功或结果与原始相同，尝试下一个策略
                logging.debug(f"  增强未生成有效新文本 - 类别: '{label}', 原始: '{text}' (策略: {name})")
            
            strategy_idx += 1
            # 如果所有策略都试过一轮且没有新增样本，可能陷入死循环，需要break
            # 这里我们循环地尝试所有策略，直到达到目标数量，所以这个上限逻辑需要调整
            # 我们可以设置一个总的尝试次数上限，或者更直接地，确保在每轮增强中，如果没能增强出新的，就继续尝试直到目标数量或者达到某个尝试上限
            # 简化逻辑，如果连续多次尝试未能生成新文本，则认为该样本难以增强
            if strategy_idx % len(strategies) == 0 and not augmented: # 尝试完一轮策略，但augmented还是空的，说明很难增强
                 logging.warning(f"类别 '{label}' 尝试一轮增强策略未能生成任何有效新文本，可能样本太短或差异性不足。")
                 if len(augmented) + ori_num < target_num: # 确保不会因为所有样本都尝试完了而停止增强
                     continue # 继续尝试下一轮，可能换个随机选择的样本就有效果了
                 else:
                     break # 已经达到目标，或者没法再增强了，就退出

        logging.info(f"类别 '{label}' 增强完成，新增 {len(augmented)} 条增强样本。")
        return samples + augmented

    def augment(self):
        logging.info("开始执行数据增强...")
        dist = self.analyze_distribution() # 原始分布
        
        # 确定目标数量
        if self.max_per_class is None:
            if not dist: # 处理空数据情况
                logging.warning("无数据可供增强。")
                self.aug_data = []
                return
            max_num = max(dist.values())
            logging.info(f"最大类别数量为: {max_num}")
        else:
            max_num = self.max_per_class
            logging.info(f"指定每个类别的最大数量为: {max_num}")

        all_augmented_samples = []
        for label, samples in self.label2data.items():
            # 确保每个类别都增强到 max_num (或其自身最大数量，如果指定了max_per_class)
            current_target_num = max_num # 固定目标为最大类数量
            # If max_per_class is not None, use it. Otherwise, use the max_num derived from distribution.
            # The `target_num` in `augment_class` should always be `max_num` for balancing.

            aug_samples = self.augment_class(label, samples, current_target_num)
            all_augmented_samples.extend(aug_samples)
            
        # 合并原始数据和所有增强数据，进行最终去重
        final_data_to_deduplicate = []
        final_data_to_deduplicate.extend(self.data) # 先加入原始数据
        final_data_to_deduplicate.extend(all_augmented_samples) # 再加入所有增强数据

        logging.info(f'去重前总样本数: {len(final_data_to_deduplicate)}')
        
        unique_samples = {}
        for item in final_data_to_deduplicate:
            key = item['text'] + '||' + str(item['text_label'])
            if key not in unique_samples:
                unique_samples[key] = item
        self.aug_data = list(unique_samples.values())
        logging.info(f'去重后总样本数: {len(self.aug_data)}')
        logging.info("数据增强完成。")

    def save_json(self):
        logging.info(f"正在保存增强数据到: {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for item in self.aug_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"增强数据保存完成。")

    def run(self):
        self.load_data()
        ori_dist = self.analyze_distribution()
        self.print_distribution(ori_dist, '原始类别分布')
        self.augment()
        aug_dist = self.analyze_distribution(self.aug_data)
        self.print_distribution(aug_dist, '增强后类别分布')
        self.save_json()
        
def main():
    # example_data_path = 'data/train.jsonl'
    # outfile = "data/train_aug_new.json"

    example_data_path = "data/processed/train_aug_cat8k_splice2.jsonl" # 替换为你想要的输出文件路径
    outfile = "data/processed/train_aug_cat8k_splice2_eda1w.json"
    
   
    augmentor = BalancedDataAugmentor(
        input_path=example_data_path,
        output_path=outfile,
        max_per_class=10000,  # 自动均衡到最大类
        seed=42
    )
    augmentor.run()

if __name__ == '__main__':
    main() 