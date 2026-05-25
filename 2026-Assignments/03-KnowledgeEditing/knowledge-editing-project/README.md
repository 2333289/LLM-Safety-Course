# 基于 EasyEdit 的大模型知识编辑实验

学生：SX2516020 陈奕晨

本项目对应课程方向 03：Knowledge Editing。代码覆盖 baseline 测试、ROME 单条事实编辑、MEMIT 批量编辑和 ES/PS/NS 指标评估。

## 目录结构

```text
.
├── baseline.py                 # 编辑前推理基线
├── edit_rome.py                # ROME 单条事实编辑
├── edit_memit.py               # MEMIT 批量事实编辑
├── evaluate.py                 # ES / PS / NS 评估
├── prepare_memit_data.py       # 生成 500 条批量编辑数据
├── data/
│   ├── custom_facts.json       # 10 条 Task 1/2 数据
│   └── memit_500_synthetic.json
├── configs/                    # EasyEdit hparams 起始配置
├── outputs/                    # 实验输出
└── reports/
    └── SX2516020-陈奕晨-03-KnowledgeEditing.md
```

## 环境准备

建议使用新的 conda 环境。当前机器上已创建并验证环境 `x1x_LLM_Safety_Work`，关键版本为 Python 3.12.13、PyTorch 2.5.1+cu121、Transformers 4.51.3。GPU 验证可识别两张 NVIDIA GeForce RTX 3090，本实验使用 GPU 0。

通用创建方式如下：

```bash
conda create -n x1x_LLM_Safety_Work python=3.10 -y
conda activate x1x_LLM_Safety_Work
pip install -r requirements.txt
```

安装 EasyEdit：

```bash
pip install easyeditor --no-deps
pip install higher hydra-core datasets gpustat einops timm iopath fairscale opencv-python-headless scipy scikit-learn sentence-transformers matplotlib peft
```

如果网络稳定，也可以安装官方仓库版本：

```bash
git clone --depth 1 https://github.com/zjunlp/EasyEdit.git
cd EasyEdit
pip install -e .
```

如果只想先验证本项目脚本、数据格式和评估流程，可以使用 mock 后端，不需要下载模型。

## 运行流程

生成 500 条 MEMIT 批量编辑数据：

```bash
python prepare_memit_data.py --output data/memit_500_synthetic.json --num-items 500
```

编辑前基线：

```bash
python baseline.py \
  --data data/custom_facts.json \
  --output outputs/baseline_predictions.jsonl \
  --backend hf \
  --model Qwen/Qwen2.5-0.5B-Instruct

python evaluate.py \
  --data data/custom_facts.json \
  --predictions outputs/baseline_predictions.jsonl \
  --output outputs/baseline_metrics.json
```

ROME 单条编辑：

```bash
python edit_rome.py \
  --data data/custom_facts.json \
  --output outputs/rome_predictions.jsonl \
  --metrics-output outputs/rome_easyedit_metrics.json \
  --backend easyedit \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hparams configs/ROME/qwen2.5-0.5b.yaml

python evaluate.py \
  --data data/custom_facts.json \
  --predictions outputs/rome_predictions.jsonl \
  --output outputs/rome_metrics.json
```

MEMIT 批量编辑：

```bash
python edit_memit.py \
  --data data/memit_500_synthetic.json \
  --output outputs/memit_predictions.jsonl \
  --metrics-output outputs/memit_easyedit_metrics.json \
  --backend easyedit \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hparams configs/MEMIT/qwen2.5-0.5b.yaml

python evaluate.py \
  --data data/memit_500_synthetic.json \
  --predictions outputs/memit_predictions.jsonl \
  --output outputs/memit_metrics.json
```

## 无 GPU / 无模型时的流程验证

```bash
python prepare_memit_data.py
python baseline.py --backend mock
python evaluate.py --data data/custom_facts.json --predictions outputs/baseline_predictions.jsonl --output outputs/baseline_metrics.json
python edit_rome.py --backend mock
python evaluate.py --data data/custom_facts.json --predictions outputs/rome_predictions.jsonl --output outputs/rome_metrics.json
python edit_memit.py --backend mock --limit 500
python evaluate.py --data data/memit_500_synthetic.json --predictions outputs/memit_predictions.jsonl --output outputs/memit_metrics.json
```

mock 模式只用于验证工程流程，正式报告中的最终指标应以 `--backend hf/easyedit` 的真实模型输出为准。

## 当前验证状态

在 `x1x_LLM_Safety_Work` 环境中已验证：

```bash
python -m py_compile baseline.py edit_rome.py edit_memit.py evaluate.py prepare_memit_data.py src/io_utils.py src/model_utils.py
python -c "from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams; print('easyedit imports ok')"
python baseline.py --backend mock --output outputs/env_baseline_predictions.jsonl
python edit_rome.py --backend mock --output outputs/env_rome_predictions.jsonl
python edit_memit.py --backend mock --limit 500 --output outputs/env_memit_predictions.jsonl
```

真实 GPU 实验已经完成，输出文件如下：

| 实验 | 样本数 | ES | PS | NS | 文件 |
|:--|--:|--:|--:|--:|:--|
| Baseline | 10 | 0.00% | 0.00% | 50.00% | `outputs/baseline_metrics_real.json` |
| ROME | 10 | 80.00% | 60.00% | 40.00% | `outputs/rome_metrics_real.json` |
| MEMIT | 500 | 0.00% | 0.00% | 0.00% | `outputs/memit_metrics_real.json` |

对应预测文件为 `outputs/baseline_predictions_real.jsonl`、`outputs/rome_predictions_real.jsonl` 和 `outputs/memit_predictions_real.jsonl`。ROME 耗时约 69.39 秒；MEMIT 500 条耗时约 1943.40 秒，记录到的 CUDA 峰值显存约 2433.64 MB。
