# VoiceAgent

**本地全双工语音训练推理内核 · 热插拔 Pi `/voice` 语音扩展 · 桌面算法实验观测工作台**

VoiceAgent 是一个聚焦 **流式语音识别（ASR）微调、流式语音合成（TTS）音色适配、全双工因果轮次预测（Turn Adapter）及消费级 8 GB GPU 低延迟推理优化** 的语音算法研发与集成项目：

1. **底层语音训练与实时推理内核（`inference/`, `training/`, `evaluation/`）**：在本地 RTX 5070 8GB 显卡上同卡运行 `Silero VAD` + 流式 `Nemotron-0.6B / Whisper` + 冻结特征驱动的 `CausalTurnGRU`（~215K 参数）+ 流式 `Qwen3-TTS-0.6B`（实测四模型峰值显存仅 **2,402.4 MB**），并提供基于 Google Drive / Hugging Face `streaming=True` 的零本地囤积微调与评测流水线。
2. **热插拔 Pi `/voice` 语音扩展（`packages/pi-voice/`）**：不重复造 Agent Runtime 轮子，以标准 `pi-package` 形态一行命令挂载至本机二次开发的 **Pi Agent**，提供像 AGY / Codex 语音模式一样的全双工免提对话、按住说话（`Alt+V`）、实时打断（Barge-in）与真实已听播放进度上下文同步。
3. **桌面算法实验与观测工作台（`apps/desktop/`）**：面向语音算法迭代的 Electron 实时观测与评测工具台——支持旁路实时绘制正在运行的 Pi `/voice` 会话波形、VAD 与 `CausalTurnGRU` 三头概率曲线，支持一键热插拔模型 Checkpoint、ASR/TTS 多模型 A/B 双盲 MOS 对比以及 Bad Case 录音注入回放。

---

## 三支柱系统架构

```text
┌────────────────────────────────────────────────────────────────────────────┐
│            本机二次开发 Pi Agent  ◄──热插拔挂载──►  packages/pi-voice        │
│       (/voice 开关 · Alt+V 快捷键 · 流式文本净化分句 · 播放游标打断对齐)       │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │ ws://127.0.0.1:8765/ws/v1/voice
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│             Python 本地语音训练与实时推理内核 (inference / training / eval)    │
│  • 在线全双工：WASAPI 原生音频守护 + AEC ──► Silero VAD ──► 流式 ASR          │
│               ├──► Partial/Final 转写                                       │
│               └──► 冻结多层 Encoder 特征 ──► CausalTurnGRU (EOT/打断/附和)    │
│               句级可读文本 ──► 流式 Qwen3-TTS ──► 24kHz 播放队列与硬件游标     │
│  • 离线微调：Google Drive / HF streaming=True ──► ASR LoRA & TTS 音色微调    │
└───────────────────────────────────────▲────────────────────────────────────┘
                                        │ Observer 旁路遥测 & REST A/B 控制
┌───────────────────────────────────────┴────────────────────────────────────┐
│                桌面算法实验与观测工作台 (apps/desktop · Electron)             │
│   实时 VAD/GRU 三头概率示波器 │ Checkpoint 热切换 │ A/B 盲测台 │ BadCase 回放 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 启动本地 Python 语音内核
```powershell
.\.venv\Scripts\python.exe -m uvicorn inference.server:app --host 127.0.0.1 --port 8765
```

### 2. 将 `/voice` 挂载到本机二次开发 Pi Agent
```powershell
# 持久路径软链挂载（后续代码改动在 Pi 内执行 /reload 即可热生效）
pi install d:/voice-agent/packages/pi-voice

# 启动 Pi 并开启语音模式
pi --voice
```

### 3. 打开桌面算法实验与观测工作台
双击桌面 `VoiceAgent` 图标，或在仓库根目录执行：
```powershell
npm run build -w @voice-agent/desktop; .\node_modules\electron\dist\electron.exe apps/desktop
```

---

## 仓库目录结构

```text
VoiceAgent/
├── inference/               # 本地语音推理与 WASAPI 音频守护内核 (VAD / ASR / TurnGRU / TTS)
├── training/                # 云端流式训练与微调流水线 (ASR LoRA / Turn Adapter / TTS Speaker)
├── evaluation/              # 模块指标评测、全双工场景回放与 GPU 性能基准套件
├── packages/
│   ├── pi-voice/            # 热插拔 Pi Package 语音扩展 (/voice 命令、快捷键与宿主生命周期钩子)
│   ├── protocol/            # WebSocket 消息信封、二进制帧编解码与三级 ID 防串台校验
│   ├── speech/              # 语音服务客户端、可读文本净化器与自适应句级缓冲器
│   ├── interaction/         # 混合轮次状态机、6 类交互裁决与三账本播放进度对齐
│   ├── metrics/             # 端到端分阶段延迟、RTF 与 GPU 显存遥测采集
│   ├── audio/               # WebAudio 采集与播放组件 (服务桌面工作台回放与试听)
│   └── agent/               # 离线单测与工作台独立冒烟用轻量 Mock Agent 桩
├── apps/
│   └── desktop/             # 桌面算法实验与观测工作台 (Electron + React + TypeScript)
├── configs/                 # 运行时与实验超参数配置
├── models/                  # 本地模型权重缓存与 models/registry.json 注册表 (忽略大权重文件)
└── docs/                    # 项目文档中心
    ├── live/                # [权威依据] V2.0 PRD 与现行全套技术/协议/实验规范
    ├── experiments/         # RTX 5070 实测基准、Turn Adapter 消融报告与 JSON 快照
    └── archive/v1/          # 已归档的 V1.0 独立桌面助手阶段历史文档
```

---

## 文档导航

### 现行权威文档（`docs/live/`）

| 文档 | 内容说明 |
|---|---|
| **[产品需求文档 (PRD V2.0)](docs/live/VoiceAgent_PRD.md)** | 三支柱重构定位、6 类全双工轮次交互、FR-01~FR-20 需求矩阵、四大微调与推理优化实验体系 |
| **[技术栈与性能预算规范](docs/live/TECH_STACK.md)** | Python WASAPI 双模式音频决策、Win CUDA 12.8 vs WSL2/AutoDL 环境分工、RTX 5070 显存实测与 TTS 加速路线 |
| **[模块功能与接口设计](docs/live/MODULES.md)** | `packages/pi-voice` 扩展契约、共享 TS 核心包、桌面实验工作台四大面板与 Python 训练推理接口 |
| **[实时通信与实验协议](docs/live/PROTOCOL.md)** | `Controller + Observer` 双角色 WebSocket 协议、三级 ID 隔离、REST 模型热插拔与 A/B 盲测接口 |
| **[开发与微调实验指南](docs/live/DEVELOPMENT_AND_EXPERIMENTS.md)** | 本地命令速查、Pi 插件软链热挂载、Google Drive / HF `streaming=True` 数据规范与 AutoDL 微调清单 |
| **[文档中心总索引](docs/README.md)** | 按研发任务划分的阅读路径与四大架构红线速查 |

### 实验报告与历史归档

- **实测实验报告（`docs/experiments/`）**：
  - [RTX 5070 8GB 四模型同卡真实闭环实测报告](docs/experiments/rtx5070_real_models_benchmark.md)
  - [CausalTurnGRU 轮次预测基线对比与消融实验报告](docs/experiments/turn_adapter_results.md)
  - [本地推理管线优化前后性能基准](docs/experiments/rtx5070_perf_benchmark.md)
- **历史归档（`docs/archive/`）**：
  - [V1 归档说明与版本演进对照表](docs/archive/README.md)

---

## 许可证与合规声明

本仓库源码采用 [MIT License](LICENSE) 开源。所引用的第三方预训练模型（NVIDIA Nemotron、OpenAI Whisper、Qwen3-TTS、Silero VAD）、宿主扩展接口（Pi Coding Agent）及公开研究数据集各自遵循其原始许可证条款。
