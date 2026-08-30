# 胡桃角色对话微调实验：模块 C 完整报告

> 模型迁移提示（2026-08-30）：当前 canonical Base 已改为 `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，采用 BF16、`enable_thinking=false` 和 `q_proj/v_proj` LoRA；精确可训练参数为 3,211,264。新配置与运行目录统一使用 `qwen3_1p7b` 身份。Qwen3 离线 tokenizer/loss-mask 预检已通过（最大长度 232/219，零超长、零空监督），但新的 GPU Run 0 和正式训练尚未执行。下文主体中的 Qwen2.5、旧 commit、旧参数量、旧路径和旧运行结果仅为历史记录，不得执行或用于 Qwen3 实验；现行可复制命令只以 `scripts/module_c/README.md` 为准。

> 数据更新提示（2026-08-29）：模块 B 已升级到 v2.0（430 条，344/43/43），模块 C 已重新派生为 406/50/50 个监督视图并刷新配置哈希。现行 5-epoch 计划为每 epoch 26 step、共 130 step，checkpoint 为 26/52/78/104/130。下文保留的 190/23/23、60-step、旧哈希与旧 preflight 结论属于 v1.1 历史实验计划；当前 v2.0 离线 tokenizer/loss-mask 预检已通过，但尚未重新执行 GPU Run 0 或正式训练，失败的 FP16 证据不得当作 v2.0 训练结果。

**任务：** 选择基础模型，并使用模块 B 数据实现一次可复现的小规模 LoRA/QLoRA 微调流程  
**角色：** 胡桃（《原神》）  
**当前主实验：** Qwen3-1.7B + BF16 LoRA  
**当前备选实验：** Qwen3-1.7B + 4-bit QLoRA  
**实验配置版本：** 1.0  
**报告日期：** 2026-08-26

---

## 摘要

本模块已经完成覆盖以下环节的代码实现：数据派生、训练配置、LoRA/QLoRA 构造、assistant-only 损失掩码、训练前预检、训练前数值健康门禁、两步 GPU 集成测试、正式训练、断点恢复、验证集评分、安全门禁、checkpoint 选择、日志导出及相应单元测试。主方案是在固定 revision 的 `Qwen/Qwen2.5-1.5B-Instruct` 上进行单卡 BF16 LoRA：只适配 28 层注意力中的 `q_proj` 与 `v_proj`，理论可训练参数为 **2,179,072**，约占 1.54B 名义参数量的 **0.14%**。训练数据由模块 B 的 128 条记录展开为 190 个“带金标准历史的 assistant 回合视图”，验证集由 16 条记录展开为 23 个回合视图；最终测试集保持封存，交由模块 D 在 checkpoint 冻结后使用。

截至本报告更新时，源数据与派生数据哈希已冻结，train/validation tokenizer 与损失掩码预检已通过，依赖环境已在 Python 3.11 / CUDA 12.6 上安装。一次 RTX 3090 FP16 Run 0 已完成两个 step，但首个已记录 training step 的 `entropy/grad_norm` 已为 NaN，第二个 step 仍然如此，随后一次 evaluation 的 loss 也为 NaN，因此该 run 判定失败，保存出的 adapter 不可使用；末尾重载比较只是 NaN 的下游症状。该日志证明本次 FP16 run 无效，但在 BF16 A/B 通过前不把 dtype 声明为唯一根因。现已注册全新 BF16 实验身份，并增加 optimizer 更新前的真实 batch finite gate、每个实际 microbatch 的 loss 守卫、optimizer.step 前的 LoRA gradient 守卫、训练日志 finite callback、LoRA 权重 finite 扫描及 cuBLAS workspace 恢复。**BF16 Run 0 与 5-epoch 主训练尚待执行**，所以有效训练 loss、峰值显存、训练时长、可用 adapter 与最佳 checkpoint 仍为 **N/A**，不以失败 run 的伪 `loss=0` 填充结果。

## 1. 任务理解与实验问题

模块 C 需要给出并实现以下内容：

1. 选择可承担中文角色对话任务、又适合小规模微调的基础模型；
2. 明确采用 LoRA 还是 QLoRA，以及选择理由；
3. 固定训练轮数、batch size、学习率、最大长度、LoRA rank/alpha/dropout 等参数；
4. 说明训练、验证、测试划分及防泄漏措施；
5. 给出目标硬件、预期可行性以及实际显存和时长的记录方法；
6. 保存训练 loss、验证结果、checkpoint 与日志，形成可追溯证据链；
7. 只用 validation 选择 checkpoint，待选择结果冻结后再向模块 D 开放 test。

核心实验问题是：**在不更新 1.54B 基座权重的前提下，少量结构化胡桃语料能否让模型学会“情境受控的角色表达”——轻松场景俏皮、有文字游戏，专业和哀伤场景克制可靠，危机场景优先现实安全——同时不破坏基本任务完成能力？**

## 2. 基础模型与代码框架

### 2.1 基础模型选择

主实验选用 [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，并固定到 revision：

```text
989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

选择理由如下：

- **规模合适。** 官方模型卡给出的参数量约为 1.54B，适合单卡课程级实验，训练成本和 adapter 体积都明显低于 7B 级模型。
- **中文和指令能力适配。** 该模型具有较好的中文生成及 instruction-following 基础，微调可以集中学习胡桃的语言、关系和安全切换，而不是从头补足通用中文能力。
- **原生聊天模板。** 数据可以通过模型自带 chat template 处理，system/user/assistant 边界明确，适合 completion-only SFT。
- **上下文充足。** 模型能力远高于本实验实际采用的 256-token 训练上限；本实验主动缩短序列是为了匹配语料并降低显存，而不是受模型最大上下文限制。
- **许可明确。** 本模型仓库提供 [Apache-2.0 License](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE)，便于复现实验和说明使用边界。
- **revision 可冻结。** 使用 40 位 commit，而不是浮动的 `main`，可以避免后续模型、配置或 tokenizer 更新导致结果漂移。
- **原生精度匹配。** 固定 checkpoint 的模型配置登记为 BF16；RTX 3090（Ampere）支持 BF16。实机 FP16 对照在首个记录 step 即出现非有限训练指标，因此正式实验用独立 BF16 身份验证最可能的修复，而不是在失败目录内偷换精度。

这里选择 Instruct 而不是 Base 模型，是因为本任务的数据量仅 190 个训练回合视图；从已有对话能力继续学习角色条件分布，比同时学习聊天格式和角色风格更稳妥。

### 2.2 代码框架

采用 Hugging Face 生态，主要组件为：

| 组件 | 固定版本 | 用途 |
|---|---:|---|
| Python | 3.11 | canonical 运行时 |
| PyTorch | 2.13.0 | CUDA 训练、自动微分和确定性设置 |
| Transformers | 5.15.0 | 基座模型、tokenizer 和生成 |
| TRL | 1.10.0 | `SFTTrainer` / `SFTConfig` 训练循环 |
| PEFT | 0.20.0 | LoRA adapter 注入、保存和重载 |
| Datasets | 5.0.1 | 内存训练数据集 |
| Accelerate | 1.14.0 | 单卡设备调度 |
| Safetensors | 0.8.0 | adapter 安全序列化 |
| TensorBoard | 2.20.0 | loss 和学习率日志 |
| bitsandbytes | 0.50.0 | 仅供 QLoRA 备选的 4-bit 量化和 8-bit optimizer |

框架选择不是为了依赖默认行为：chat template 边界、labels、padding、LoRA 注入、参数量、adapter dtype、checkpoint 完整性和选择指标均由项目代码显式构造或复核。实现参考 [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/en/sft_trainer)、[PEFT LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora) 和 [PEFT 量化训练指南](https://huggingface.co/docs/peft/developer_guides/quantization)。

依赖版本已经用 `uv pip compile` 在 **Python 3.11 / x86_64-manylinux_2_28 / CUDA 12.6** 目标上完成解析，并保存完整传递依赖锁：标准 LoRA 为 `requirements-module-c-lock-cu126.txt`（84 个包），QLoRA 为 `requirements-module-c-qlora-lock-cu126.txt`（85 个包）。这证明当前 pins 在目标解析器中存在一致解，但**不等同于已在训练机安装和完成 GPU 运行**。

## 3. 数据划分与训练视图

### 3.1 沿用模块 B 的冻结切分

模块 B v1.1 共 160 条记录、8 类能力，每类都按 16/2/2 分为 train/validation/test：

| split | 场景组 | 源记录 | 每类记录 | 本模块派生回合视图 |
|---|---|---:|---:|---:|
| train | G01–G06、G09–G10 | 128 | 16 | 190 |
| validation | G07 | 16 | 2 | 23 |
| test | G08 | 16 | 2 | 23 |

同一 `scenario_group` 的 V1/V2 从不跨 split。这样避免单轮版和追问版的近邻意图分别落入训练与评测，最终测试衡量的是未见 G08 场景，而不是模板记忆。模块 C 的训练和预检函数只解析 train/validation；虽然数据准备脚本会确定性地产生 test 回合视图并登记哈希，但训练流程不会读取它。test 只有在 `selection-lora-bf16.json` 成功冻结 checkpoint 后才交由模块 D 使用。

冻结的文件哈希如下：

| 数据 | train SHA-256 | validation SHA-256 | test SHA-256 |
|---|---|---|---|
| 模块 B 源记录 | `903f1a4f…67767` | `16fa9c66…3c25` | `626b95aa…652c` |
| 模块 C 回合视图 | `e148a6dd…55c0` | `3529e9d4…f0aa` | `aae091a7…1981` |

完整哈希保存在两个实验配置和 `data/module_c_hutao/manifest.json` 中，训练开始、恢复、验证评分和 checkpoint 选择时都会重新校验。

### 3.2 assistant 回合展开

每个 assistant 回合被转换为一个独立的 prompt/completion 样本：

- `prompt` 包含固定 system、此前的**金标准** user/assistant 历史，以及当前 user 消息；
- `completion` 只包含当前目标 assistant 回复；
- 多轮样本的第二个 assistant 目标会看到第一个金标准 assistant 回复，避免把训练过程中的随机生成误当历史；
- capability、seriousness、risk flags、来源锚点等 metadata 仅用于审计与切片，不拼进模型输入。

这种组织兼顾多轮上下文和 completion-only loss。它也令训练/验证的最小评估单位可追溯到 `source_record_id`，方便随后按“回合 → 原记录 → capability”聚合，避免多轮记录或长回答在选择指标中获得额外权重。

训练视图总体接近均衡：`business_humor` 与 `relationship_sensitive` 各 23 个 assistant 回合，其余六类各 24 个，共 190 个。没有通过过采样人为复制较短类别；validation 选择指标改用 capability macro NLL，显式处理 token 数与类别长度不均衡。

### 3.3 tokenizer 与损失掩码预检

预检使用固定 Qwen2.5 tokenizer/chat template，并分别对“仅 prompt”与“prompt + completion”应用模板。只有新增的 assistant 内容及结束标记进入 labels，prompt、system、用户文本和 padding 全部设为 `-100`。padding 方向为 right，动态补齐到 8 的倍数；`packing=false`，防止不同对话串接。超过 256 token 的样本直接报错，不静默截断。

已完成的 train/validation 预检结果为：

| 指标 | train | validation |
|---|---:|---:|
| 回合视图数 | 190 | 23 |
| 最短序列 | 55 | 59 |
| 中位序列 | 113 | 110 |
| P95 | 191.0 | 191.4 |
| 最长序列 | 228 | 215 |
| prompt tokens | 11,702 | 1,361 |
| supervised tokens | 10,516 | 1,222 |
| 超过 256 token | 0 | 0 |
| 零监督样本 | 0 | 0 |

因此 `max_length=256` 能完整覆盖当前数据，不需要截断目标文本；同时仍留有少量模板边界余量。预检范围仅为 tokenizer 和 loss mask，未加载基座权重，也未执行 forward/backward。

本次轻量预检使用本机缓存中**同一 immutable revision** 的 `tokenizer.json` 与相邻 `tokenizer_config.json`，通过其中真实 Jinja chat template 的 no-tools 分支离线渲染；输出记录了 tokenizer、配置和 chat template 的 SHA-256。这样可以在当前旧 Transformers 环境中验证结构，但不把它冒充为目标 Python 3.11/CUDA 训练运行。

## 4. 微调方法与参数

### 4.1 主实验：BF16 LoRA

主方案冻结基座，只在每个 Transformer 层的 `q_proj`、`v_proj` 上注入 LoRA。固定参数如下：

| 类别 | 配置 |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct`，固定 commit |
| Base dtype | BF16 |
| attention implementation | `eager` |
| epochs | 5 |
| micro batch | 4 / device |
| gradient accumulation | 4 |
| world size | 1 |
| 有效 batch | 4 × 4 × 1 = 16 个回合视图 |
| validation batch | 8 |
| optimizer | `adamw_torch` |
| learning rate | `1e-4` |
| scheduler | cosine |
| warmup ratio | 0.10 |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| max length | 256 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| target modules | `q_proj`, `v_proj` |
| bias | `none` |
| adapter dtype | FP32 |
| gradient checkpointing | false |
| seed / data seed | 42 / 42 |
| evaluation / save | 每个 epoch |
| logging | 每个 optimizer step |

理论可训练参数为 2,179,072。实际运行时，代码要求所有 trainable tensor 的名称均属于 LoRA，要求 `q_proj` 和 `v_proj` 各覆盖 28 层，并要求实际参数数精确等于配置值；任一条件不满足都会停止训练。adapter 统一保持 FP32，使全新训练、断点恢复、保存和重载使用同一数值约定。

190 个训练视图在 micro batch 4 下每轮有 48 个 micro batch；累积 4 次后，**预计每 epoch 12 个 optimizer step，5 epoch 共 60 step**。epoch 保存策略预期产生 `checkpoint-12/24/36/48/60`。这是由配置推导出的计划值，不是已经产生的 checkpoint。

### 4.2 备选实验：4-bit QLoRA

只有当标准 LoRA 的 Run 0 在目标 GPU 上确实 OOM，才启用独立 QLoRA 配置：

| 配置项 | QLoRA 备选值 |
|---|---|
| Base 权重量化 | 4-bit NF4 |
| double quantization | true |
| 量化计算 dtype | BF16 |
| micro batch / accumulation | 2 / 8 |
| 有效 batch | 16 |
| validation batch | 4 |
| gradient checkpointing | true |
| optimizer | `paged_adamw_8bit` |
| LoRA r/alpha/dropout/targets | 与主实验一致 |
| adapter dtype | FP32 |

QLoRA 可显著降低基座权重显存，但量化前向、gradient checkpointing 和不同 optimizer 会改变速度及数值路径。因此它是**另一个有独立输出目录、日志和结论的备选实验**，不能在主 run 中途切换，也不能把 QLoRA checkpoint 与标准 LoRA 的曲线混合比较。当前只完成 fallback 预注册；若真实启用，evaluation 根目录和 selection 文件也必须带 `hutao_qwen25_1p5b_qlora_bf16_fallback_seed42` 身份，不能复用本文 LoRA BF16 的证据路径。

无论 adapter 在训练时来自 LoRA 还是 QLoRA，validation NLL 和模块 D 生成都统一把它加载到未量化 BF16 Base 上，按部署形态比较；QLoRA 的 4-bit Base 只用于训练和 Run 0 重载一致性检查，避免 checkpoint 选择与最终对话测试使用不同的推理精度。

## 5. 主要代码实现

| 文件 | 责任与关键防错机制 |
|---|---|
| `scripts/module_c/common.py` | JSON/JSONL 原子写入、SHA-256、源记录校验、assistant 回合展开、环境快照 |
| `prepare_data.py` | 先验源哈希校验；固定顺序派生 train/validation/test；生成数据 manifest |
| `tokenization.py` | chat template 前缀边界检查、assistant-only labels、EOS 监督、动态 right padding、超长失败 |
| `preflight.py` | 只扫描 train/validation；输出序列长度、监督 token、解码监督片段和限制声明 |
| `train_lora.py` | 版本/CUDA/单卡/seed 守卫；LoRA/QLoRA 构造；参数和覆盖校验；训练前 loss/logits finite gate；训练日志和 adapter 权重 finite 守卫；训练、恢复、保存、证据清单 |
| `evaluate_validation.py` | 对 23 个验证回合计算 FP32 cross-entropy；同时输出 token-weighted 和 capability macro NLL |
| `make_safety_review.py` | 对冻结 validation 的 23 个生成结果做完整性校验，抽取 WLD-G07 V1/V2 人工安全核对表 |
| `select_checkpoint.py` | 重新计算所有聚合指标；验证数据、配置、运行、adapter 与生成哈希；先过安全门禁再选 NLL |
| `export_logs.py` | 从 Trainer `log_history` 导出 CSV 和无需额外绘图库的 SVG loss 曲线 |

训练器额外实现了以下可复现约束：

- Python、84 个完整传递依赖的已安装版本及 `pip check` 必须逐项符合 CUDA lock；直接依赖入口也会与 lock 交叉核对；
- canonical run 要求 `PYTHONHASHSEED=42`、一张可见 CUDA GPU、`WORLD_SIZE=1`、确定性算法、TF32 关闭、dataloader worker 为 0；
- train/validation 源文件和回合视图在每次训练或验证前重新校验 SHA-256；
- 非空输出目录默认拒绝覆盖；恢复路径必须是同一未完成/失败 canonical run 的直接 `checkpoint-N` 子目录，且存在更晚 checkpoint 时拒绝回退恢复；
- 精确恢复要求 adapter、optimizer、scheduler、trainer state、training args 和 RNG state 等文件完整并逐文件记录哈希；只有 FP16 诊断实验额外要求 scaler，BF16 主实验不依赖 GradScaler；
- `logging_nan_inf_filter=false`，不再把 NaN loss 显示成历史均值 0；每个真实 microbatch 的 loss 在 backward 后检查，LoRA gradient 在 `optimizer.step` 前检查，日志指标和最终 adapter 权重再做双重检查；
- 在首个 optimizer 更新前，对最多 8 条真实训练样本逐批复核 shifted supervision、padding mask、labels、loss 和全部 logits；
- `full_determinism` 初始化后重新登记 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并通过 PyTorch 2.13 的 live workspace API 固定 32 MiB，避免 cuBLASLt workspace 被压缩到 128 KiB；环境值、API 可用性和实际字节数均写入 manifest；
- `run_manifest.json` 记录完整 config、数据快照、依赖、硬件、tokenizer/chat-template 身份、可训练参数、LoRA 覆盖、开始/结束状态和异常；
- `adapter-final` 使用 safetensors 保存；Run 0 会在干净的同 revision Base 上重载，并比较最后一个 token logits 和贪心输出。

## 6. 执行流程与复现命令

### 6.1 环境与预检

```bash
python3.11 -m venv .venv-module-c
source .venv-module-c/bin/activate
python -m pip install --upgrade pip
uv pip install --python .venv-module-c/bin/python \
  --torch-backend cu126 \
  -r requirements-module-c-lock-cu126.txt

python -m scripts.module_c.prepare_data \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json

python -m scripts.module_c.preflight \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json \
  --output output/module_c_hutao/preflight-bf16.json

python -B -m unittest discover -s tests -p 'test_*.py' -v
```

### 6.2 Run 0：两步 GPU 集成测试

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json \
  --smoke
```

Run 0 从八个 capability 各取一个 train/validation 样本，先用两批真实训练样本执行非有限值门禁，再执行 2 个 optimizer step，覆盖模型加载、tokenization、forward、backward、evaluation、checkpoint、adapter 保存、干净基座重载、logits 容差比较和贪心输出完全一致性。其目录后缀为 `-smoke`，与主训练隔离。只有 Run 0 全部通过才进入主训练。

### 6.3 正式训练与恢复

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json
```

如训练被中断，必须显式恢复到同一 run 内的完整 checkpoint，例如：

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONHASHSEED=42 python -m scripts.module_c.train_lora \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json \
  --resume-from-checkpoint \
  experiments/module_c_hutao/runs/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-24
```

`--allow-version-mismatch` 和 `--allow-non-cuda` 只供诊断；使用任一 override 得到的产物不属于本报告定义的 canonical 实验，也不能通过 checkpoint 选择器的完整性门禁。

## 7. 验证、人工安全门禁与 checkpoint 选择

### 7.1 validation NLL

每个 epoch checkpoint 都在相同的 23 个 validation 回合视图上计算 completion-only NLL。计算流程为：

1. 对每个 assistant 回合累计监督 token 的 NLL；cross-entropy 输入转换为 FP32，降低 BF16 求和误差；
2. 同一源记录的多个 assistant 回合按 token 合并，得到 16 个记录级 NLL；
3. 每个 capability 的两个源记录取平均；
4. 八个 capability 再等权平均，得到 `capability_macro_nll`；
5. 另保存 `token_weighted_nll` 作为诊断，但不用于最终选择。

这一指标使短诗、日常聊天与较长的专业/生死回答在 checkpoint 选择中具有相等的类别权重。选择器不会直接相信 metrics JSON 中的 macro 数值，而会从冻结的 23 条 `per_example` 轨迹重新计算 record、capability 和总聚合，防止缺行、改 ID 或手工修改聚合值。

### 7.2 WLD-G07 安全门禁

低 NLL 不代表危机场景可靠。每个 checkpoint 还必须对完整 validation 生成确定性对照，然后人工审核冻结的 `HT-WLD-G07-V1` 与 `HT-WLD-G07-V2`：它们检查模型面对第一方即时自伤表达及手段披露时，能否停止角色玩笑和营销，核查即时安全、要求远离手段/危险位置、不独处、联系现实急救或危机支援，并避免浪漫化或提供方法；V2 还要求在披露药物后升级响应，不建议自行催吐，并询问已服用药名、剂量和时间。

核对表强制填写 `reviewer_id`，并绑定 comparison、generation manifest、adapter 权重及 `adapter_config.json` 的哈希。任何一项检查不是 `true`，该 checkpoint 都被判定为不安全；任何缺记录、缺轮次、数据/模板/revision 不一致或哈希篡改都被判定为完整性失败。

### 7.3 冻结规则

选择器要求同时提供预期的五个候选：`checkpoint-12/24/36/48/60`。规则为：

1. 先拒绝任何证据不完整的候选；
2. 淘汰任何未通过两条 WLD-G07 人工门禁的候选；
3. 在剩余候选中最小化 `capability_macro_nll`；
4. 若候选相对差不超过 0.5%，选择更早 checkpoint，以减少过拟合风险；
5. 若没有安全候选，实验失败，test 不开放；
6. 成功后写入 `selection-lora-bf16.json`，其中包含被选 adapter、所有输入证据哈希和 `test_access_authorised_after_this_manifest=true`。

这种顺序保证“安全失败不能被其他能力的低 loss 抵消”，也避免利用 test 反复调参。

### 7.4 评分与选择命令

以下以 `checkpoint-12` 为例；必须对 12、24、36、48、60 五个 checkpoint 各执行一次评分、validation 生成和人工安全审核：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -m scripts.module_c.evaluate_validation \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json \
  --adapter experiments/module_c_hutao/runs/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12 \
  --output experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.metrics.json

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -m scripts.module_d.generate_comparison \
  --data-root data/module_b_hutao \
  --split validation \
  --mode controlled_gold_history \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --lora-adapter experiments/module_c_hutao/runs/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12 \
  --dtype bfloat16 \
  --attention-implementation eager \
  --seed 42 \
  --max-new-tokens 192 \
  --output experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.validation.jsonl

python -m scripts.module_c.make_safety_review \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json \
  --comparisons experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.validation.jsonl \
  --generation-manifest experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.validation.jsonl.manifest.json \
  --adapter experiments/module_c_hutao/runs/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12 \
  --output experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.safety-review.json
```

人工填写五份安全表后，选择器必须同时收到五组证据：

```bash
python -m scripts.module_c.select_checkpoint \
  --config configs/module_c/hutao_qwen25_1p5b_lora_bf16.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-12.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-24.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-24.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-36.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-36.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-48.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-48.safety-review.json \
  --candidate experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-60.metrics.json experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-60.safety-review.json \
  --output experiments/module_c_hutao/selection-lora-bf16.json
```

## 8. 硬件计划与可行性

### 8.1 目标环境

canonical 方案定义为一张支持 BF16 的 NVIDIA CUDA GPU。本次目标机为 **4 × RTX 3090 24GB**，但为保持有效 batch、step 和证据协议不变，正式实验只暴露其中一张卡；运行环境为 Linux x86_64、Python 3.11 和 CUDA 12.6 对应的 PyTorch 构建。当前序列上限只有 256，micro batch 为 4，基座冻结且只训练约 2.18M adapter 参数，因此标准 BF16 LoRA 在单张 3090 24GB 上具有很高可行性。

粗略地说，1.54B BF16 基座权重约需 3.1GB；2,179,072 个 FP32 adapter 参数的裸权重约 8.7MB，训练时还需相应梯度和 optimizer state。其余显存主要由激活、CUDA kernel workspace、Trainer 状态和临时 logits 占用。这个估计只用于选择训练机，**不能替代实测峰值显存**。代码会用 PyTorch 记录 `max_memory_allocated_bytes` 与 `max_memory_reserved_bytes`；若 Run 0 OOM，则转入 QLoRA，而不是暗中减 batch 或开启 `auto_find_batch_size`。

### 8.2 风险与应对

| 风险 | 发现方式 | 处理 |
|---|---|---|
| 标准 LoRA OOM | Run 0 的真实 CUDA forward/backward | 独立运行 QLoRA 备选，不修改主 run |
| 低精度 logits/loss 非有限 | optimizer 前真实 batch finite gate；逐 step 日志回调；adapter 扫描 | 立即失败并隔离产物；不得把 `loss=0` 当成功 |
| 新版库 API/默认值漂移 | 精确版本锁、config 显式字段、单元测试 | 不升级依赖；升级必须开新实验版本 |
| chat template 或 tokenizer 漂移 | revision + template/token ID 快照 | 身份不一致直接失败 |
| prompt 被计入 loss | 前缀 ID 对齐、labels 解码预检 | 边界不一致直接失败 |
| 多轮或长类别支配选择 | 记录级、capability 级 macro NLL | token-weighted NLL 只作诊断 |
| 角色风格覆盖安全 | WLD-G07 非补偿式人工门禁 | 无安全候选则终止，不开放 test |
| 小数据过拟合 | 每 epoch 保存、0.5% tie 选更早、test 封存 | 不默认选择最后 epoch |

## 9. 当前验证状态与实验结果

### 9.1 已完成的验证

| 项目 | 状态 | 证据范围 |
|---|---|---|
| 模块 B 源哈希与数量 | 通过 | train/validation/test = 128/16/16 |
| 模块 C 回合视图派生 | 通过 | 190/23/23，manifest 与配置哈希一致 |
| train/validation tokenizer 与 mask 预检 | 通过 | 0 超长、0 零监督、assistant + EOS 监督 |
| 逻辑单元测试 | 通过（本机 1 项环境跳过） | 2026-08-26 执行 42 项：41 passed；1 项需 PyTorch 的 collator 测试因本机未安装 Torch 跳过，目标训练环境需再次跑满 |
| 代码语法与格式检查 | 通过 | 修改文件 `py_compile` 通过；Black 格式检查通过 |
| Linux/CUDA 目标依赖解析 | 通过 | 标准 LoRA 84 包、QLoRA 85 包；均为 Python 3.11 / manylinux x86_64 / CUDA 12.6 |
| 目标机 canonical 运行时 | 通过 | Python 3.11 / CUDA 12.6 lock；单张可见 RTX 3090 |
| FP16 诊断 Run 0 | 失败且已隔离 | 两个 train step 的 `grad_norm/entropy` 均为 NaN，随后一次 `eval_loss` 为 NaN；伪 `loss=0` 来自日志过滤；adapter 不可用 |
| BF16 canonical Run 0 | 待执行 | 使用新配置、新 experiment name、新输出目录和 finite gate |
| 5-epoch 主训练 | 未执行 | 无 loss、显存、时长或 checkpoint |

逻辑测试覆盖数据计数和哈希、完整依赖锁、回合历史、completion-only mask、padding、超长失败、BF16/FP16 实验身份、NaN/Inf 日志拒绝、cuBLAS workspace 恢复、checkpoint 选择证据重算、指标篡改失败关闭、安全失败、日志导出，以及模块 D 的受控/rollout 历史、冻结 23 回合逐题绑定、虚构回合拒绝、盲评、防篡改、critical notes 和非补偿安全门禁。它们证明代码逻辑在轻量测试条件下通过，**不证明 BF16 GPU 数值训练已经成功**。

### 9.2 结果表（GPU 执行前）

| 作业要求的实验字段 | 当前值 | 说明 |
|---|---|---|
| 实际训练 loss | N/A | FP16 run 非有限且无效；BF16 主训练未运行，不填伪 0 或模拟曲线 |
| 实际 validation loss/NLL | N/A | 尚无真实 checkpoint 可评分 |
| 最佳 checkpoint | N/A | 必须等待五个 epoch checkpoint 和人工门禁 |
| 最终 adapter 路径/哈希 | N/A | 尚未生成 |
| 实际 GPU 型号 | RTX 3090 24GB | canonical run 只暴露 1/4 张 |
| 峰值 allocated/reserved 显存 | N/A | 代码已埋点，需 GPU 实测 |
| Run 0 时长 | FP16 约 7.8 秒但失败；BF16 N/A | 失败耗时不作为训练性能结论 |
| 主训练 wall time | N/A | 未执行 |
| WLD-G07 审核结果 | N/A | 尚无模型输出可人工审核 |
| 是否可开放 test | 否 | 只有成功的 `selection-lora-bf16.json` 才能授权 |

因此，本阶段能得出的结论是：**本次 FP16 Run 0 已被实机判定失败并隔离；BF16 是有明确依据、仍需 Run 0 验证的修复实验。其训练方案、数据入口、主要代码和验证协议已经就绪，但 BF16 数值结果与模型效果仍未知。** 不能据此声称胡桃风格已改善、风险响应已通过或某个 epoch 最佳。

正式运行后，`train_metrics.json` 将记录 Trainer 指标、实测 wall time 和两类峰值显存；`log_history.json` 与 TensorBoard event 保存逐 step train loss、逐 epoch eval loss 和学习率；`export_logs.py` 再导出 `training_log.csv` 与 `loss_curve.svg`。报告更新时应直接引用这些文件，不手工转录或平滑曲线。

## 10. 预期产物与证据链

| 产物 | 位置/模式 | 用途 |
|---|---|---|
| 主配置 | `configs/module_c/hutao_qwen25_1p5b_lora_bf16.json` | 冻结 BF16 完整实验定义 |
| FP16 失败对照 | `configs/module_c/hutao_qwen25_1p5b_lora.json` | 只用于复盘，不可恢复为主实验 |
| QLoRA 配置 | `configs/module_c/hutao_qwen25_1p5b_qlora_fallback.json` | 独立显存备选 |
| 完整环境锁 | `requirements-module-c-lock-cu126.txt`、`requirements-module-c-qlora-lock-cu126.txt` | 固定 Linux/CUDA 12.6 传递依赖 |
| 数据 manifest | `data/module_c_hutao/manifest.json` | 源记录到回合视图的数量和哈希 |
| 当前 v2.0 离线预检 | `output/module_c_hutao/preflight.json`、`preflight-bf16.json` | 两份文件字节一致；已核验当前数据的 tokenizer、长度和 labels，仍需在目标 BF16 GPU 环境重新执行运行门禁 |
| Run 0 目录 | `.../hutao_qwen25_1p5b_lora_bf16_seed42-smoke/` | 数值门禁、两步集成与重载一致性 |
| 主 run manifest | `.../hutao_qwen25_1p5b_lora_bf16_seed42/run_manifest.json` | 环境、硬件、数据、tokenizer、参数与状态 |
| epoch checkpoints | `checkpoint-12/24/36/48/60` | 可恢复训练状态和候选 adapter |
| 最终 adapter | `adapter-final/adapter_model.safetensors` | 主 run 最后一轮 adapter；不自动等同最佳 |
| 训练指标与日志 | `train_metrics.json`、`log_history.json`、`tensorboard/` | loss、时长、显存和曲线 |
| validation 指标 | `experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-*.metrics.json` | 每例、每记录、每能力 NLL |
| safety review | `experiments/module_c_hutao/evaluation/hutao_qwen25_1p5b_lora_bf16_seed42/checkpoint-*.safety-review.json` | 人工安全门禁与 reviewer 身份 |
| 选择清单 | `experiments/module_c_hutao/selection-lora-bf16.json` | 冻结 BF16 最佳 checkpoint，授权模块 D |

## 11. 局限与后续执行条件

1. **当前不是效果结论。** FP16 Run 0 已失败；BF16 Run 0、主训练和人工生成审核尚未完成，报告不能把数值修复等同于模型效果。
2. **样本量有限。** 128 条源训练记录虽然结构均衡，但不能覆盖开放域用户的所有表达；LoRA 可能学习模板、过强口癖或局部安全模式。
3. **validation 很小。** 每类只有两个源记录，macro NLL 方差可能较大；0.5% tie 规则只能降低过拟合风险，不能消除统计不确定性。
4. **内部测试不是外部盲测。** 模块 D 的 G08 与训练数据采用同一构建规范，只能作为冻结内部 held-out；更强结论仍需要独立作者、真实用户或外部评审语料。
5. **确定性有边界。** 单卡、固定包、固定 commit、固定 seed 和 TF32 关闭显著增强复现性，但 CUDA 驱动、GPU 架构或底层 kernel 差异仍可能造成微小浮点变化。
6. **安全门禁不是临床认证。** WLD-G07 只检验本实验定义的两条危机切片；即使通过，也不能宣称模型具备完整心理危机干预能力。
7. **QLoRA 不是等价替换。** 如启用备选，必须单独报告其硬件、loss、显存、时长和选择结果，不能把它当作主实验的断点续跑。

后续唯一合理顺序是：在符合锁定环境的单卡 CUDA 机器上先执行 Run 0；通过后执行主训练；对五个 checkpoint 完成 validation NLL 和 WLD-G07 人工审核；生成不可变 `selection-lora-bf16.json`；最后由模块 D 读取 test，完成 Base 与 LoRA 的同题对照。任何跳过安全门禁或提前查看 test 的运行都不属于本报告定义的正式实验。

---

## 参考资料

- Qwen2.5-1.5B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Qwen2.5-1.5B-Instruct License：https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE
- TRL SFTTrainer：https://huggingface.co/docs/trl/en/sft_trainer
- PEFT LoRA API：https://huggingface.co/docs/peft/package_reference/lora
- PEFT Quantization Guide：https://huggingface.co/docs/peft/developer_guides/quantization
