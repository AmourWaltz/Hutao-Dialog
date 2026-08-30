# 模块 C：Qwen3-1.7B 胡桃 LoRA/QLoRA 实验

主实验固定为 `Qwen/Qwen3-1.7B` BF16 LoRA：

- revision：`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；
- chat template：`{"enable_thinking": false}`，训练、验证和生成必须一致；
- tokenizer 的 `bos_token_id=null` 是 Qwen3 的合法身份值；不得人工改成 PAD/EOS，校验器会区分显式 `null` 与字段缺失；
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`，覆盖 28 层的 `q_proj`、`v_proj`；
- 精确可训练参数：`3,211,264`；adapter 保持 FP32；
- 数据：344/43/43 条源记录，406/50/50 个监督目标；
- 5 epoch，每 epoch 26 optimizer step，共 130 step。

旧的 `hutao_qwen25_1p5b_lora.json` 是 Qwen2.5 FP16 失败诊断，只作历史证据，训练入口会拒绝执行。Qwen3 的 4-bit QLoRA 是独立 fallback，不得与标准 LoRA 共用目录或 checkpoint。

## 1. 环境

使用 Linux x86_64、Python 3.11、单张支持 BF16 的 CUDA GPU。正式环境安装完整 CUDA 12.6 lock：

```bash
python3.11 -m venv .venv-module-c
source .venv-module-c/bin/activate
python -m pip install --upgrade pip
uv pip install --python .venv-module-c/bin/python \
  --torch-backend cu126 \
  -r requirements-module-c-lock-cu126.txt
```

锁定的 `transformers==5.15.0` 高于 Qwen3 所需的 4.51.0。canonical run 要求 `WORLD_SIZE=1` 且只暴露一张 GPU。

## 2. 数据派生与预检

```bash
python -m scripts.module_c.prepare_data \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json

python -m scripts.module_c.preflight \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --output output/module_c_hutao/preflight.json

python -B -m unittest discover -s tests -p 'test_*.py' -v
```

预检验证 Qwen3 non-thinking 模板的 prefix/full 边界、assistant-only labels、EOS、空监督和 256-token 上限。当前离线实测 train/validation 最大长度为 232/219，零超长、零空监督。预检不加载模型权重，也不替代 GPU Run 0。

## 3. Run 0：两步 GPU smoke

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --smoke
```

Run 0 输出到：

```text
experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42-smoke/
```

它会验证依赖锁、CUDA/BF16、模型 commit、chat template、3,211,264 个 LoRA 参数、28 层覆盖、有限 loss/gradient 和 checkpoint。重载门禁会先移除 Accelerate 的训练期 autocast forward 包装，使保存前模型与干净重载走同一推理路径；随后核对 adapter 配置，并对 112 个 FP32 LoRA 张量逐值执行零容差回环校验，同时要求贪心 token IDs 完全一致。BF16 全词表 logits 差异只作为诊断记录。目录非空时脚本拒绝覆盖；失败证据应改名归档后再重新运行，不能使用 `--resume-from-checkpoint` 恢复一个尚无有效 checkpoint 的下载失败。

## 4. 正式训练与恢复

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json
```

主输出目录：

```text
experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/
```

断点恢复示例：

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --resume-from-checkpoint experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-52
```

恢复时会重新核对配置哈希、数据快照、checkpoint 状态和 Base revision。Qwen2.5 checkpoint/adapter 不能用于 Qwen3。

## 5. Validation、人工安全门禁与 checkpoint 选择

本节仅用于严格实验流程。普通课设可完全跳过本节，直接使用训练目录中的
`adapter-final`，并按模块 D README 的 `--use-final-adapter` 命令运行 test；该路径
不会读取 validation metrics、人工 safety-review 或 selection manifest。

对每个 `checkpoint-26/52/78/104/130` 执行一次 validation NLL：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -m scripts.module_c.evaluate_validation \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --adapter experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26 \
  --output experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.metrics.json
```

生成同 checkpoint 的 validation 对照时必须显式关闭 thinking：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -m scripts.module_d.generate_comparison \
  --data-root data/module_b_hutao \
  --split validation \
  --mode controlled_gold_history \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --lora-adapter experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26 \
  --dtype bfloat16 \
  --attention-implementation eager \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --seed 42 \
  --max-new-tokens 192 \
  --output experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.validation.jsonl

python -m scripts.module_c.make_safety_review \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --comparisons experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.validation.jsonl \
  --generation-manifest experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.validation.jsonl.manifest.json \
  --adapter experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26 \
  --output experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.safety-review.json
```

人工完成五份安全表后，将五组 metrics/review 传给选择器：

```bash
python -m scripts.module_c.select_checkpoint \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-26.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-52.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-52.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-78.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-78.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-104.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-104.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-130.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen3_1p7b_lora_bf16_seed42/checkpoint-130.safety-review.json \
  --output experiments/module_c_hutao/selection-qwen3-1p7b-lora-bf16.json
```

先淘汰未通过安全门禁的候选，再最小化九类 capability 的记录级 macro NLL；相对差不超过 0.5% 时选择更早 checkpoint。五个 epoch checkpoint 缺一不可。

## 6. 日志

```bash
python -m scripts.module_c.export_logs \
  --log-history experiments/module_c_hutao/runs/hutao_qwen3_1p7b_lora_bf16_seed42/log_history.json \
  --csv output/module_c_hutao/training_log_qwen3_1p7b_bf16.csv \
  --svg output/module_c_hutao/loss_curve_qwen3_1p7b_bf16.svg
```

在新 Qwen3 GPU Run 0 和正式训练完成前，不应填写 loss、峰值显存、训练时长、最佳 checkpoint 或效果结论。
