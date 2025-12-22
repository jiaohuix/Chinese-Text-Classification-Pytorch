# 🚀 BERT 文本分类全链路加速方案：从 QLoRA 到 OpenVINO 极致部署

本项目记录了针对 **THUNews** 新闻分类任务的完整性能优化路径。通过将 **参数高效微调 (PEFT)** 与 **Intel 硬件原生加速 (OpenVINO)** 结合，我们在普通的 10 核 CPU 上实现了对比 PyTorch 原生推理 **13.4 倍** 的吞吐量提升。

## 1. 项目背景与技术选型

在工业化部署中，BERT-Base（12层）通常面临推理延迟大的问题。本项目采用了以下技术栈：

* **微调阶段**：使用 **QLoRA (4-bit)**。在极低显存占用下，通过训练小型 Adapter 达到全量微调精度。
* **合并阶段**：将 Adapter 与原模型合并，恢复为标准 FP16 结构，为后续量化做准备。
* **量化阶段**：利用 **OpenVINO NNCF** 工具。通过 PTQ (训练后量化) 将模型压缩至 INT8。
* **硬件加速**：针对 Intel **AVX-512 VNNI** 指令集优化，实现 CPU 级别的矩阵运算加速。

## 2. 📁 项目结构与配置驱动

项目采用 **Config-Driven** 设计，只需修改 YAML 配置文件，即可实现不同训练/推理模式的无缝切换。

```text
project/
├── configs/
│   ├── fp16.yaml         # 全量微调配置
│   ├── qlora.yaml        # 4-bit Adapter 微调配置
│   ├── ptq_int8.yaml     # OpenVINO INT8 量化导出配置
│   └── ptq_int4.yaml     # OpenVINO INT4 量化导出配置
├── train.py              # 统一训练入口 (支持全量/QLoRA)
├── eval.py               # PyTorch 原生评估脚本 (用于基准测试)
├── eval_ov.py            # OpenVINO 专属评估脚本 (多流并行压测)
├── merge_lora.py         # LoRA 权重合并工具
├── export_ov.py          # 模型量化转换工具
└── utils.py              # 硬件检测、Logger配置与解析

```

## 3. 🚀 环境安装与数据预处理

```bash
# 1. 基础环境与 OpenVINO 加速组件安装
pip install transformers datasets peft accelerate evaluate pyyaml tqdm
uv pip install "optimum-intel[openvino,nncf]"

# 2. 数据准备：克隆并预处理 THUNews
git clone https://github.com/jiaohuix/Chinese-Text-Classification-Pytorch.git
cd Chinese-Text-Classification-Pytorch && git checkout dev && cd multi_class
python scripts/thunews_preprocess.py 

```

## 4. 🏃‍♂️ 操作流水线

### 4.1 模型训练与基准评估 (PyTorch)

```bash
# 1. 训练：根据 config 自动选择 FP16 或 QLoRA
python train.py --config configs/qlora.yaml

# 2. 评估：测试 FP16/LoRA 模型在 CPU 上的原始表现
python eval.py -c configs/fp16.yaml -m experiments/bert_v1/best_model -d cpu -n 1000 -b 8

```

### 4.2 权重合并

完成 QLoRA 训练后，必须将分布式存储的 Adapter 注入回基座模型，形成独立的 FP16 文件供 OpenVINO 转换。

```bash
python merge_lora.py \
    --adapter_path experiments/bert_v1/best_adapter \
    --save_path experiments/bert_v1/best_adapter_merged

```

### 4.3 OpenVINO 导出与极致压测

将合并后的模型转化为针对 Intel 硬件优化的 IR 格式。

```bash
# 1. 导出量化模型 (推荐 INT8)
python export_ov.py -m experiments/bert_v1/best_adapter_merged -o experiments/bert_v1/ov_int8 -q int8

# 2. 开启多流自动调度 (重要！)
export OV_CPU_THROUGHPUT_STREAMS=CPU_THROUGHPUT_AUTO

# 3. 运行 10000 条大规模压测
python eval_ov.py -c configs/fp16.yaml -m experiments/bert_v1/ov_int8 -b 8 -n 10000

```

## 5. 📊 实验报告 (硬件: 10核 Intel Xeon Platinum 8350C)

| 评测指标 | PyTorch 原生 (FP16) | OpenVINO (INT8) | OpenVINO (INT4) |
| --- | --- | --- | --- |
| **吞吐量 (TPS)** | 25.24 samples/s | **339.84 samples/s** | 96.50 samples/s |
| **准确率 (Acc)** | 0.9400 (小样) / 0.8981 | **0.8912** | 0.8900 |
| **单条延迟** | ~40 ms | **~2.9 ms** | ~10.3 ms |
| **提升倍数** | 基准 | **13.4 倍 🚀** | 3.8 倍 |

### 🔍 核心洞见 (Insights)

1. **INT8 优于 INT4 的真相**：在支持 **AVX-512 VNNI** 的 CPU 上，INT8 有原生硬件加速；而 INT4 需额外的软件解压开销，在计算密集型模型（如 BERT）中反而更慢。
2. **多核并行与预热**：在 10000 条大规模压测下，OpenVINO 的 TBB 线程池调度更充分，吞吐量远超小规模测试。
3. **架构演进 (UIE)**：由固定分类转向“双指针 Head (Start/End)”，配合 Prompt 实现 0-shot 动态标签提取。

## 📝 TODO 列表 (Next Steps)

* [ ] **模型保存增强**：在训练逻辑中加入 `id2label` 和 `label2id` 的自动保存，解决推理时的映射依赖。
* [ ] **代码解耦**：将 `utils.py` 中的 Logger 重构为可传参对象，减少硬编码。
* [ ] **QAT 实验**：引入 `NNCF` 的真正的量化感知训练 (Quantization Aware Training)，替代现有的 PTQ 以追求无损精度。
* [ ] **知识蒸馏**：使用 **TextBrewer** 训练 6 层 TinyBERT，利用老师模型对齐 Attention 层，冲击 600+ TPS。
* [ ] **C++ 部署**：编写基于 OpenVINO C++ Runtime 的推理后端，消除 Python 解释器瓶颈。

---

**归档日期**：2025-12-21

**项目状态**：已打通全量数据评估链路，INT8 方案达到工业级部署标准。
