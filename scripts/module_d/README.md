# 模块 D：Base 与胡桃 LoRA 的测试对比

课设默认使用模块 C 训练结束时保存的 `adapter-final`，不需要 checkpoint
选择、validation metrics 或人工安全门禁。主分析采用 `controlled_gold_history`：
Base 与 LoRA 在每个 assistant 目标处收到完全相同的 system/user/金标准历史，
用来隔离当前回答差异；`rollout` 让两模型使用各自前序生成，仅作为多轮误差
累积诊断。当前 test 含 43 条源记录和 50 个监督目标，覆盖 9 个 capability。

## 1. 直接使用最终 adapter 生成 test 对照

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python -m scripts.module_d.generate_comparison \
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

`--use-final-adapter` 不读取任何 checkpoint metrics、safety-review 或 selection
manifest。它只做运行所需的轻量一致性检查：

- adapter 目录确实是完成训练后写出的 `adapter-final`；
- 相邻 `run_manifest.json` 的状态为 `complete`，其中路径和权重 SHA-256
  与当前 adapter 一致；
- `adapter_config.json`、训练 manifest 和命令使用同一个 Base model/revision；
- Base 与 LoRA 使用同一个 Base 路径、revision、tokenizer/chat template 和解码配置；
- Qwen3 全链路固定 `enable_thinking=false`，避免思考段进入回答、rollout 历史或盲评表；
- seed、`max_new_tokens`、BF16 dtype 与 eager attention 必须等于模块 C 注册值；
- 输出必须与冻结源数据重建出的 50 个监督目标逐项一致，不能漏题或用虚构回合补数；导入多轮记录只评最终回复，通用桥接回合保留为金标准历史；
- 输出 manifest 记录源 test、训练 run manifest、final adapter 和生成结果哈希。

原有严格流程仍兼容：如确实需要 validation checkpoint 选择，可不传
`--use-final-adapter`，改传 `--selection-manifest` 和其中选中的 `checkpoint-N`。

可用同样参数另跑 `--mode rollout`，输出到独立文件。不要把 rollout 与 controlled 的分数混在一个主表中。

## 2. 自动规则 + 模型裁判

自动路径对三个角色层各计算 3 个指标：

| 层级 | 规则指标（0–100） | 模型指标（1–5） | 层内权重 |
|---|---|---|---|
| 角色表层说话风格 | `style_marker_control`、`syntax_register_match` | `model_layer_score` / `surface_style` | 30% / 30% / 40% |
| 知识关系层 | `factual_support`、`relationship_constraints` | `model_layer_score` / `knowledge_relationship` | 35% / 35% / 30% |
| 价值观/世界观层 | `principle_action_coverage`、`conflict_safety` | `model_layer_score` / `value_worldview` | 35% / 25% / 40% |

三层总分权重为 30% / 35% / 35%。模型裁判分按 `(score - 1) × 25` 转为 0–100 后与规则分聚合。五个维度的完整 1–5 分档位只维护在 `scripts/module_d/rubric.py`，并同时用于模型裁判和人工盲评。

模型裁判默认配置为 DeepSeek V4 Pro：

- API model：`deepseek-v4-pro`；
- 当前登记版本：`DeepSeek-V4-Pro-0813`；
- 官方地址：`https://api.deepseek.com`；
- `thinking=disabled`、`temperature=0`、JSON Output；
- API Key 仅从环境变量读取，不写入命令参数或评测产物。

每条 DeepSeek 请求都会完整携带三层指标的 definition、indicators 和
1–5 全部打分锚点，并显式告诉裁判：目标角色固定为胡桃，要评价的是
“候选回答本身”在说话风格、知识关系和价值观/世界观上接近胡桃的程度。
裁判理由保持客观中文，不让裁判自己扮演胡桃，避免用角色口吻掩盖评分依据。

首先生成盲化 judge 请求和隔离 key。模型和版本默认从配置读取，无需再填写
占位符：

```bash
python -m scripts.module_d.evaluate_automatic prepare \
  --comparisons experiments/module_d_hutao/test-controlled.jsonl \
  --generation-manifest experiments/module_d_hutao/test-controlled.jsonl.manifest.json \
  --config configs/module_d/hutao_three_layer_eval.json \
  --seed 42 \
  --requests-jsonl experiments/module_d_hutao/test-controlled.judge.requests.jsonl \
  --key-json experiments/module_d_hutao/test-controlled.judge.key.json
```

该显式胡桃目标指令属于 `module_d.judge_request.v2`。如果之前已生成过
v1 requests/key，必须重新执行 `prepare`；已有评分或 audit 时，请先归档旧文件
或给新一轮使用新文件名，不要混用旧提示产物。

随后在 Bash 中安全输入 Key；`read -s` 不会把 Key 回显到屏幕或写进命令历史：

```bash
read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

调用 DeepSeek V4 Pro 完成 50 条裁判请求：

```bash
python -m scripts.module_d.run_deepseek_judge \
  --requests-jsonl experiments/module_d_hutao/test-controlled.judge.requests.jsonl \
  --key-json experiments/module_d_hutao/test-controlled.judge.key.json \
  --config configs/module_d/hutao_three_layer_eval.json \
  --output-jsonl experiments/module_d_hutao/test-controlled.judge.scored.jsonl \
  --audit-json experiments/module_d_hutao/test-controlled.judge.deepseek.audit.json \
  --api-key-env DEEPSEEK_API_KEY
```

Runner 原样保留每行其他字段，只填写 `judgment`；API request ID、模型返回名、
`system_fingerprint`、token usage 和重试记录保存在独立 audit JSON 中，
DeepSeek API Key 不会写入任何评测产物。
每条有效结果都会立即落盘；网络中断后重复同一命令即可从已完成题目续跑。
调用时会把盲化后的上下文和 A/B 回答发送到 DeepSeek API；如数据不允许
交给外部服务，不应运行这一步。

Runner 完成后可以从当前 shell 移除 Key：

```bash
unset DEEPSEEK_API_KEY
```

A/B 的 `surface_style`、`knowledge_relationship`、`value_worldview`、
`task_completion`、`safety_ethics` 都必须是 1–5 整数分、非空理由和至少一段来自
对应回答的逐字证据。Runner 会本地检查；返回内容不合格时，使用同一份冻结提示
最多重试 3 次，不会追加会改变裁判口径的“纠错提示”。

这里的 `seed=42` 只固定题目顺序和 A/B 盲化；DeepSeek Chat API 当前没有
`seed` 请求参数，因此不能宣称模型输出可逐 token 复现。`deepseek-v4-pro` 是官方
服务别名，配置中的 `DeepSeek-V4-Pro-0813` 是本次登记版本而不是可传给 API 的
不可变 revision；Runner 会记录返回的 `system_fingerprint`，并在同一次评测中发现
指纹漂移时停止，避免把不同后端版本的分数混在一起。

全部请求完成后汇总：

```bash
python -m scripts.module_d.evaluate_automatic score \
  --judge-results experiments/module_d_hutao/test-controlled.judge.scored.jsonl \
  --judge-audit experiments/module_d_hutao/test-controlled.judge.deepseek.audit.json \
  --key-json experiments/module_d_hutao/test-controlled.judge.key.json \
  --config configs/module_d/hutao_three_layer_eval.json \
  --summary-json experiments/module_d_hutao/test-controlled.automatic.summary.json
```

汇总前会重新校验 comparison、generation manifest、模型/revision、seed、rubric/config/prompt/request 哈希、题数、A/B 回答与逐字证据，并要求每条评分都能在 DeepSeek audit 中找到哈希一致的成功 API 调用。输出分开保留 `Base`、`LoRA`、`Delta` 和简化的 `before_after`；规则/模型分与人工分不混合。

DeepSeek V4 的模型名和 JSON Output 调用方式参见官方文档：
[API 快速开始](https://api-docs.deepseek.com/)；
[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。

## 3. 构建人工盲评表

```bash
python -m scripts.module_d.build_review_sheet \
  --comparisons experiments/module_d_hutao/test-controlled.jsonl \
  --generation-manifest experiments/module_d_hutao/test-controlled.jsonl.manifest.json \
  --review-csv experiments/module_d_hutao/test-controlled.human.blind.csv \
  --key-json experiments/module_d_hutao/test-controlled.human.key.json \
  --rubric-json experiments/module_d_hutao/test-controlled.human.rubric.json \
  --seed 42
```

只把 CSV 和公开 rubric JSON 交给评审者，key JSON 必须隔离。A/B 顺序和题目顺序都由固定 seed 打乱；评分脚本会拒绝被修改的题目、上下文或模型回答。

每个 A/B 回答都使用与模型裁判完全相同的 1–5 分档位，分别评价三层角色分 `surface_style`、`knowledge_relationship`、`value_worldview`，以及两个独立门禁 `task_completion`、`safety_ethics`。角色总分是三层等权平均，门禁不计入该总分。

另填：

- `critical_failure_a/b`：严重安全事故、核心身份/关系颠倒、可造成显著现实后果的严重任务失败或模板/特殊 token 泄漏；
- `error_tags_a/b`：以逗号分隔，只能使用 rubric 登记的预注册标签；
- `surface_style_preference`、`knowledge_relationship_preference`、`value_worldview_preference` 和总体 `preference`：均填 `A`、`B` 或 `Tie`；
- `reviewer_id`：必填；
- `notes`：简述关键判断；任一侧为 critical failure 时强制必填。

最低要求是完成一份覆盖全部 50 题的有效盲评表。若使用多名评审者，应让其分别填写独立副本，再对三层相差至少 2 分、critical failure 不一致或总体偏好相反的项目做第三人复核。

## 4. 汇总人工评分

```bash
python -m scripts.module_d.score_review \
  --scored-csv experiments/module_d_hutao/test-controlled.human.blind.csv \
  --key-json experiments/module_d_hutao/test-controlled.human.key.json \
  --output experiments/module_d_hutao/test-controlled.human.summary.json
```

输出包含逐题解盲分数、三层及角色总分的 `before_after`、Base/LoRA/差值、两个门禁、9 个 capability 分层、分层与总体 win/tie/loss、critical failure 与错误标签频数。

课设简化模式不设置 validation 安全门禁，也不以安全审核结果筛选 checkpoint。
报告重点比较人工三层角色总分、各层 Base/LoRA 差值和非平局胜率；
`task_completion`、`safety_ethics` 与 critical failure 仍可作为诊断信息列出，但不会
阻止 test 生成或要求重新选择 checkpoint。

## 5. 报告口径

自动规则/模型裁判与人工盲评必须各自列出三层的 Base（微调前）、LoRA（微调后）和 `LoRA - Base`，不能将两类分数平均成一个总分。当前尚无真实 adapter、Base/LoRA 回答和评分，因此报告中的效果数值为 `N/A`；这是未执行状态，不是 0 分，也不表示微调前后相同。
