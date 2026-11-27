# VoiceAgent 文档中心（Documentation Index）

本目录汇总了 **VoiceAgent V2.0** 的权威产品需求基线、三支柱技术架构规范、模块接口契约、实时通信与实验协议、实测实验报告以及已归档的 V1 阶段历史文档。

---

## 1. 目录结构与职责划分

```text
docs/
├── README.md                                 # [当前文件] 文档总览与按场景阅读导航
├── live/                                     # [权威活文档基线 · 当前开发与实验唯一依据]
│   ├── VoiceAgent_PRD.md                     # 产品需求文档 V2.0 (三支柱定位、FR-01~FR-20、四大实验矩阵)
│   ├── TECH_STACK.md                         # 技术栈选型、双环境隔离、RTX 5070 8GB 显存实测与 TTS 加速路线
│   ├── MODULES.md                            # 全系统 6 大子系统职责与 TypeScript / Python 接口契约
│   ├── PROTOCOL.md                           # 本地 WebSocket (Controller + Observer) 与 REST 实验控制协议
│   └── DEVELOPMENT_AND_EXPERIMENTS.md        # 本地开发、Pi /voice 热插拔挂载、云盘流式数据与 AutoDL 微调手册
├── experiments/                              # [实测实验归档 · 不可变快照与消融报告]
│   ├── turn_adapter_results.md               # CausalTurnGRU 对比基线与层数/特征消融实验报告
│   ├── turn_adapter_manifest.json            # Turn Adapter 实验元数据快照
│   ├── rtx5070_real_models_benchmark.md      # RTX 5070 8GB 四模型同卡真实闭环实测报告 (2,402.4 MB 峰值)
│   ├── real_4model_rtx5070_manifest.json     # 四模型实测 JSON 快照
│   ├── rtx5070_perf_benchmark.md             # 推理管线优化前后延迟与显存对比报告
│   └── rtx5070_perf_manifest.json            # 管线性能实测 JSON 快照
└── archive/                                  # [历史归档 · 仅作架构演进溯源，不再作为执行依据]
    ├── README.md                             # V1 -> V2 架构重构背景与文件对照说明
    └── v1/                                   # V1.0 独立桌面语音助手阶段旧版 PRD 与技术文档
```

> **活文档治理准则**：凡涉及需求边界、算法实验矩阵、协议事件或模块接口的调整，均须优先更新 `docs/live/` 对应文档；已淘汰方案统一移入 `docs/archive/`，实测产出统一沉淀至 `docs/experiments/`。

---

## 2. 按任务场景阅读指南

### A. 开展 ASR / TTS 微调与 Turn Adapter 算法实验（核心主线）
1. 阅读 [VoiceAgent_PRD.md](live/VoiceAgent_PRD.md) 第 4–5 章，明确 `TURN-A0~A5`、`ASR-A0~A3`、`TTS-A0~A3`、`OPT-01~04` 四大实验矩阵与验收指标。
2. 阅读 [DEVELOPMENT_AND_EXPERIMENTS.md](live/DEVELOPMENT_AND_EXPERIMENTS.md) 第 3–4 章，严格遵守 **Google Drive + Hugging Face `streaming=True` 零本地数据集囤积规范** 与 AutoDL ¥100 预算防忘关机清单。
3. 对照 [experiments/](experiments/) 中的现有 RTX 5070 基线报告输出新实验对比表并更新 `models/registry.json`。

### B. 开发并热挂载 Pi `/voice` 语音扩展（TypeScript）
1. 阅读 [MODULES.md](live/MODULES.md) 第 1–2 章，掌握 `packages/pi-voice` 的 `pi-package` 清单约束（`peerDependencies` 铁律）、`ExtensionAPI` 生命周期钩子与三账本播放游标对齐逻辑。
2. 对照 [PROTOCOL.md](live/PROTOCOL.md) 实现 `role: "controller"` + `audio_mode: "local_device"` 握手、`tts.synthesize` 短句推送与 `generation.cancel` 打断协同。

### C. 升级桌面算法实验与观测工作台（Electron + React）
1. 阅读 [MODULES.md](live/MODULES.md) 第 3 章与 [PROTOCOL.md](live/PROTOCOL.md) 第 1、5 章，了解如何通过 `role: "observer"` 旁路实时绘制 Pi 对话期间的 VAD / Causal GRU 概率曲线，以及如何调用 `/v1/models/activate` 与 `/v1/ab-test/*` 完成模型热插拔与双盲 MOS 评测。

---

## 3. 四大架构红线速查

1. **零侵入复用既有 Pi Agent**：严禁在仓库内重复造完整的 Agent Runtime 轮子，日常智能体能力全部通过 `packages/pi-voice` 热插拔复用本机二次开发的 Pi Agent。
2. **数据集零本地落盘囤积**：大规模训练与评测语料一律走 Google Drive（`VOICE_AGENT_DATA_ROOT`）或 Hugging Face `streaming=True` 流式加载。
3. **在线推理与离线训练严格解耦**：训练中断或实验指标未达标的权重不得污染默认配置，仅验证达标并注册进 `models/registry.json` 的权重可供在线挂载。
4. **本地回环与默认拒绝安全边界**：语音内核仅监听 `127.0.0.1`；语音辅助工具确认在超时或语义含糊时一律按拒绝（Fail-Closed）处理。
