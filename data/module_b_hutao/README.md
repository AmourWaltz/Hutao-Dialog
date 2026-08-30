# 胡桃角色对话 SFT/LoRA 数据卡（v2.0）

## 数据集概览

v2.0 将原有 160 条人工策划样本与根目录 `all_samples.jsonl` 的 270 条样本全量合并，形成 430 条统一的 `messages` JSONL 数据。数据用于训练“现代安全适配版胡桃角色助手”：轻松场景保持俏皮和文字游戏，职业、哀伤及危机场景优先责任、边界与现实安全。

| 文件 | 记录数 | 用途 |
|---|---:|---|
| `train.jsonl` | 344 | 参数更新 |
| `validation.jsonl` | 43 | 早停、超参数与 checkpoint 选择 |
| `test.jsonl` | 43 | checkpoint 冻结后的最终评测 |
| `all.jsonl` | 430 | 排序后的完整合并集 |
| `categories/*.jsonl` | 430 | 可重建的规范化源记录 |
| `schema.json` | — | JSON Schema |
| `manifest.json` | — | 来源、切分、计数与 SHA-256 |

全集共有 170 个场景组、264 条单轮记录、166 条多轮记录，原始 user/assistant 回合各 596 个。模块 C 实际监督 506 个 assistant 目标，按 train/validation/test 为 406/50/50。

## 合并与规范化

原有 160 条记录保持内容和场景组不变。导入的 270 条记录执行以下确定性转换：

- `valid` 规范为 `validation`；
- system 消息统一为“你是胡桃，以符合角色设定且适合当前情境的方式回答。”；
- 场景组增加 `EXT-` 命名空间，避免与旧 ID 冲突；
- 原文件物理行、原 ID/group/split/category、完整非消息元数据、原 system 和记录摘要保存在 `metadata.source`；
- 9 个来源 category 映射到项目能力体系，其中 `knowledge_boundary` 作为新增能力保留；
- 导入的 V2 首轮是跨场景重复的通用过渡模板，因此保留为金标准上下文，但通过 `assistant_turn_policy=final_only` 只监督最终回复。

能力映射如下：

| `all_samples` category | 统一 capability |
|---|---|
| `business_professional` | `professional_funeral` |
| `daily_playful` | `daily_chat` |
| `grief_support` | `empathy_grief_support` |
| `knowledge_boundary` | `knowledge_boundary` |
| `life_death_values` | `worldview_life_death` |
| `liyue_relationships`、`traveler_paimon` | `relationship_sensitive` |
| `poetry_wordplay` | `wordplay_poetry` |
| `safety_crisis` | `crisis_leadership` |

合并后的 capability 分布为：

| capability | train | validation | test | 合计 |
|---|---:|---:|---:|---:|
| `business_humor` | 16 | 2 | 2 | 20 |
| `crisis_leadership` | 34 | 5 | 5 | 44 |
| `daily_chat` | 46 | 5 | 5 | 56 |
| `empathy_grief_support` | 46 | 5 | 5 | 56 |
| `knowledge_boundary` | 18 | 3 | 3 | 24 |
| `professional_funeral` | 46 | 5 | 5 | 56 |
| `relationship_sensitive` | 64 | 8 | 8 | 80 |
| `wordplay_poetry` | 28 | 5 | 5 | 38 |
| `worldview_life_death` | 46 | 5 | 5 | 56 |

## 切分与泄漏控制

切分单位始终是 `scenario_group`，同组变体不会跨 split。输入文件中还存在与旧数据 held-out 场景高度相近的概念；导入器对 14 个完整场景组做等量对换，使这些概念与旧场景处于同一 split，同时保持导入数据 216/27/27、各来源 category 的 train/validation/test 分布不变。具体覆盖清单记录在 `manifest.json` 的 `split_policy.conflict_aware_overrides`。

导入数据的 90 个 V2 桥接回合只有 8 种模板且横跨三个 split。它们不会成为监督目标，避免把重复的通用过渡句计入 validation/test。最终监督目标不存在跨场景组的标准化精确重复；同一导入场景组内的目标复用属于有意的提示变体。

仍需注意：这是合成数据，不等同于 430 个统计独立的真实用户意图。部分概念在不同来源间存在语义邻近，尤其是哀伤和危机场景；`manifest.json` 中的记录数不应被解释为独立分布样本数。

## 记录格式

每行一个对象，顶层固定为 `id`、`messages`、`metadata`：

```json
{
  "id": "hutao-daily_playful-cold_hands-v2",
  "messages": [
    {"role": "system", "content": "你是胡桃，以符合角色设定且适合当前情境的方式回答。"},
    {"role": "user", "content": "我有件事想偷偷问你。"},
    {"role": "assistant", "content": "哦？神神秘秘的，讲来听听。"},
    {"role": "user", "content": "手好冷，可我又懒得戴手套。"},
    {"role": "assistant", "content": "手指都冷得不听使唤了，还和手套闹别扭？捂暖以后乖乖戴上。"}
  ],
  "metadata": {
    "split": "test",
    "capability": "daily_chat",
    "scenario_group": "EXT-hutao-daily_playful-cold_hands",
    "assistant_turn_policy": "final_only",
    "source": {"dataset": "all_samples.jsonl", "line": 6}
  }
}
```

示例为便于阅读省略了部分必需 metadata；完整字段以 `schema.json` 为准。metadata 仅用于切分、审计和评测，不默认拼入模型输入。

## 重建与校验

从根目录执行：

```bash
python3 scripts/module_b/import_all_samples.py
python3 scripts/module_b/apply_content_audit_fixes.py
python3 scripts/module_b/build_dataset.py
python3 scripts/module_b/validate_dataset.py
python3 -m scripts.module_c.prepare_data \
  --config configs/module_c/hutao_qwen3_1p7b_lora_bf16.json
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```

导入、构建与派生过程均为确定性的；重复执行会产生相同文件哈希。当前模块 B 校验结果为 430/430 schema 通过、270/270 导入来源可回溯、0 error、0 warning。

当前核心哈希：

| 文件 | SHA-256 |
|---|---|
| `train.jsonl` | `5da8a647e0f1f52a90abb3814cbf3b1cc82a46ad146bd184e75ef7a8df91fe10` |
| `validation.jsonl` | `42562316c1a2fa3f83313154c75b08ff53b6ab5fd19526e315ba3be08cd8af0d` |
| `test.jsonl` | `d9e1f88a9ac180e2f08330f01b7093542ee6a8d59665745ad70280e1341ccf2c` |
| `all.jsonl` | `cb8b9bc0224baba5559e1121d1a452a8ddd62fcd4d299c40df4fbb817628b8f7` |

## 使用边界

- 只用 `train.jsonl` 更新参数，validation/test 不参与训练。
- 使用模型原生 chat template，并采用 assistant-only loss。
- 导入记录只监督最终 assistant 回复，但 prompt 保留完整多轮金标准历史。
- 高风险样本用于学习安全切换，不构成医疗、法律、消防、危机干预或葬仪专业服务。
- 数据为角色约束下的原创合成内容，不应据此声称获得官方角色授权或真实用户分布覆盖。
