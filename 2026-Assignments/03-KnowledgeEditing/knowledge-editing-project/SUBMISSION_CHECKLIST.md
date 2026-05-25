# 提交清单

学生：SX2516020 陈奕晨

方向：03-KnowledgeEditing

## 文件清单

| 要求 | 对应文件 | 状态 |
|:--|:--|:--|
| baseline 推理脚本 | `baseline.py` | 已完成 |
| ROME 单条编辑脚本 | `edit_rome.py` | 已完成 |
| MEMIT 批量编辑脚本 | `edit_memit.py` | 已完成 |
| 综合评估脚本 | `evaluate.py` | 已完成 |
| 依赖文件 | `requirements.txt` | 已完成 |
| 运行说明 | `README.md` | 已完成 |
| 实验报告 | `SX2516020-陈奕晨-03-KnowledgeEditing.md` | 已完成 |
| 10 条事实编辑数据 | `data/custom_facts.json` | 已完成 |
| 500 条批量编辑数据 | `data/memit_500_synthetic.json` | 已完成 |
| ROME 配置 | `configs/ROME/qwen2.5-0.5b.yaml` | 已完成 |
| MEMIT 配置 | `configs/MEMIT/qwen2.5-0.5b.yaml` | 已完成 |

## 已验证命令

```bash
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python -m py_compile baseline.py edit_rome.py edit_memit.py evaluate.py prepare_memit_data.py src/io_utils.py src/model_utils.py
MPLCONFIGDIR=/tmp /home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python -c "from easyeditor import ROMEHyperParams, MEMITHyperParams; ROMEHyperParams.from_hparams('configs/ROME/qwen2.5-0.5b.yaml'); MEMITHyperParams.from_hparams('configs/MEMIT/qwen2.5-0.5b.yaml'); print('hparams-ok')"
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python baseline.py --backend mock --output outputs/env_baseline_predictions.jsonl
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python edit_rome.py --backend mock --output outputs/env_rome_predictions.jsonl
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python edit_memit.py --backend mock --limit 500 --output outputs/env_memit_predictions.jsonl
```

## 真实 GPU 实验结果

当前机器可访问 GPU，`nvidia-smi` 显示两张 NVIDIA GeForce RTX 3090，`torch.cuda.is_available()` 为 `True`。真实实验已完成：

| 实验 | 样本数 | ES | PS | NS | 输出 |
|:--|--:|--:|--:|--:|:--|
| Baseline | 10 | 0.00% | 0.00% | 50.00% | `outputs/baseline_metrics_real.json` |
| ROME | 10 | 80.00% | 60.00% | 40.00% | `outputs/rome_metrics_real.json` |
| MEMIT | 500 | 0.00% | 0.00% | 0.00% | `outputs/memit_metrics_real.json` |

报告已如实记录真实编辑结果和失败案例分析；mock pipeline 仅作为工程流程验证保留。
