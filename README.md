# Hutao-Dialog

胡桃角色对话数据集与微调评测项目，聚焦于角色一致性、语料构建、LoRA 微调和自动/人工评测闭环。

## 项目定位

该仓库围绕“胡桃角色助手”展开，按任务阶段拆分为四个模块：

- 模块 A：角色定位与语料分析
- 模块 B：对话数据构建与清洗
- 模块 C：Qwen3 小规模 LoRA 微调实验
- 模块 D：对比测试、自动裁判与人工盲评

项目目标是：

- 形成结构化、可审计的角色对话语料；
- 训练一个遵循胡桃角色设定的对话模型；
- 通过冻结测试集和多层评测指标验证效果；
- 保留实验流程、日志和证据，确保复现和审计。

## 仓库结构

```text
.
├── README.md
├── requirements-module-c-qlora.txt
├── configs/
│   └── module_c/
├── data/
│   ├── module_b_hutao/
│   └── module_c_hutao/
├── experiments/
│   ├── module_c_hutao/
│   └── module_d_hutao/
├── output/
│   ├── docs/
│   ├── module_b_hutao/
│   ├── module_c_hutao/
│   └── module_d_hutao/
├── scripts/
│   ├── module_b/
│   ├── module_c/
│   └── module_d/
├── tests/
│   ├── test_module_c.py
│   ├── test_module_d_automatic.py
│   ├── test_module_d_deepseek.py
│   └── test_module_d.py
└── .git/
```

## 模块说明

### 模块 A：角色与语料分析

重点在于：

- 确定胡桃角色的说话风格、边界和世界观；
- 归纳语料中的能力维度；
- 形成统一的角色设计与语料分布分析。

相关输出位于：

- output/docs/
- output/module_a_hutao/

### 模块 B：数据构建

数据构建路线以 JSONL 语料为主，包含：

- 导入样本
- 审计修正
- 数据集生成
- 校验与 schema 检查

关键目录：

- data/module_b_hutao/
- scripts/module_b/

常用流程：

```bash
python3 scripts/module_b/import_all_samples.py
python3 scripts/module_b/apply_content_audit_fixes.py
python3 scripts/module_b/build_dataset.py
python3 scripts/module_b/validate_dataset.py
```

### 模块 C：LoRA / QLoRA 实验

模块 C 主要负责：

- 数据准备；
- 预检（preflight）
- 微调训练；
- checkpoint 评估；
- 安全门禁与选择

核心配置位于：

- configs/module_c/
- scripts/module_c/
- experiments/module_c_hutao/

典型命令：

```bash
python -m scripts.module_c.prepare_data \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json

python -m scripts.module_c.preflight \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --output output/module_c_hutao/preflight.json

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 \
python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json
```

### 模块 D：评测与裁判

模块 D 用于：

- 生成 Base 与 LoRA 对比结果；
- 进行自动规则评分；
- 调用 DeepSeek 等模型裁判；
- 构建人工盲评表并汇总结果。

关键目录：

- scripts/module_d/
- experiments/module_d_hutao/
- output/module_d_hutao/

典型命令：

```bash
python -m scripts.module_d.generate_comparison \
  --data-root data/module_b_hutao \
  --split test \
  --mode controlled_gold_history \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --lora-adapter experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/adapter-final \
  --use-final-adapter \
  --dtype bfloat16 \
  --attention-implementation eager \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --seed 42 \
  --max-new-tokens 192 \
  --output experiments/module_d_hutao/test-controlled.jsonl
```

## 运行环境要求

建议使用：

- Python 3.11
- Linux x86_64
- 单卡支持 BF16 的 CUDA GPU
- 具备 Hugging Face / Transformers / PEFT 等依赖

相关环境说明可参考：

- requirements-module-c-qlora.txt
- scripts/module_c/README.md
- scripts/module_d/README.md

## 测试

项目包含一组单元/集成测试，用于检查模块 C 与模块 D 的关键行为：

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

## 输出产物

本项目的核心产物包括：

- 训练与评估配置：configs/
- 语料数据：data/
- 实验运行：experiments/
- 结果与报告：output/
- 评测脚本：scripts/

其中，output 中的文档报告和评测总结是实验结论的主要归档材料。

## 注意事项

- 数据与评测流程以“固定 seed + 确定性文件哈希 + 训练/评测清单”作为约束。
- 模块 C 的主实验采用 Qwen3 1.7B 的 LoRA/BF16 路径，要求聊天模板和推理设定保持一致。
- 模块 D 中的自动评测与人工打分应分别保留独立结果，不混用。 
- 处理真实 DeepSeek 裁判时，API Key 仅从环境变量读取，不写入命令参数或产物。

## 参考文档

- scripts/module_c/README.md
- scripts/module_d/README.md
- data/module_b_hutao/README.md
- output/docs/

如果你要继续推进研发工作，可以优先按“数据构建 → 训练 → 评测 → 报告”的顺序执行。
