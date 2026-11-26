# VoiceAgent

**本地全双工桌面语音智能体 & 语音交互算法研发平台**

VoiceAgent 是一个面向 **Windows 11 + NVIDIA GPU** 的桌面级实时语音助手与算法实验平台。系统以 **Pi Agent SDK** 为任务推理与工具执行核心，采用本地流式 **Nemotron ASR** 与 **Qwen3-TTS** 构建低延迟语音链路，并以可训练的 **Causal GRU Turn Adapter + 混合轮次控制器** 实现自然的全双工对话、句中停顿保持、附和恢复与实时语音打断（Barge-in）。

同时，项目提供与在线推理相互解耦的离线训练与评测体系，支持 **ASR 微调、TTS 音色适配、轮次预测 Adapter 训练** 及端到端性能基准测试。

---

## 核心特性

- **自然全双工语音交互**：播放助手回复期间持续监听麦克风，精准区分**发言完成（EOT）、句中停顿（HOLD）、短暂附和（Backchannel）、主动打断（Interruption）与环境噪声（Noise）**，支持 Duck / Pause / Resume / Cancel 多级音频控制。
- **本地流式语音模型**：默认搭载 `nvidia/nemotron-speech-streaming-en-0.6b` 流式语音识别与 `Qwen3-TTS`（0.6B）流式语音合成，语音链路全程本地运行，无需商业语音 API。
- **Pi Agent 深度集成与安全工具调用**：直接复用 Pi SDK 的多模型接入、会话上下文与工具生态；内置 `Conversation` / `Action` 双运行模式与 `Readonly` / `Ask` / `Trusted` 三级权限网关，支持界面与语音双重授权确认。
- **基于真实播放进度的上下文同步**：实时追踪扬声器实际已播放文本边界，被打断后精确告知 Agent“用户实际听到了哪里”，同时完整保留已执行工具的真实状态。
- **可训练与可评测的算法平台**：内置冻结 ASR Encoder 多层特征提取、因果 GRU Turn Adapter 训练、ASR/TTS 微调流水线与覆盖 10 类系统级交互场景的评测套件，在 8 GB 消费级显存预算内稳定运行。

---

## 系统架构概览

```text
┌──────────────────────────────────────────────────────────┐
│              Electron + React + TypeScript GUI           │
│       Transcript / Dual-Track State / Approvals / HUD    │
└────────────────────────────┬─────────────────────────────┘
                             │
         Microphone ──► Audio Capture & 16kHz Resampler
                             │
                             ▼
             AEC (with Playback Ref) & Noise Suppression
                             │
                             ▼
                  Local WebSocket (127.0.0.1)
                             │
                             ▼
               Silero VAD + Nemotron Streaming ASR
                      ┌──────┴───────┐
                      ▼              ▼
                ASR Decoder    Causal Turn Adapter
                      │              │
                      └──────┬───────┘
                             ▼
                 Hybrid Turn Manager (TS)
                      │              │
         (Interrupt / Duck / Pause)  │ (Commit Turn)
                      │              ▼
                      │        Pi Agent SDK
                      │       ┌──────┴──────┐
                      │       ▼             ▼
                      │      LLM       Agent Tools (Guarded)
                      │       └──────┬──────┘
                      │              ▼
                      │   Text Filter & Sentence Buffer
                      │              │
                      ▼              ▼
               Streaming Qwen3-TTS ──► Audio Playback Queue
                                             │
                                             └──► AEC Reference & Playback Ack
```

---

## 技术栈速览

| 层级 | 模块 | 技术选型 |
|---|---|---|
| **桌面与应用层** | Desktop / GUI / Runtime | Electron + React + TypeScript + Node.js |
| **音频管线** | 采集 / 播放 / 前处理 | Web Audio API (`AudioWorklet`) + Ring/Pre-roll Buffer + WebRTC Audio Processing (AEC/NS) |
| **语音检测与识别** | VAD / Streaming ASR | Silero VAD v5 + NVIDIA NeMo (`nvidia/nemotron-speech-streaming-en-0.6b`) |
| **全双工轮次控制** | Turn-Taking / Barge-in | PyTorch Causal GRU Adapter + 混合轮次状态机（含 Smart Turn / 静音阈值基线） |
| **智能体核心** | Agent Runtime & Tools | Pi Agent SDK (`@mariozechner/pi-agent-core` / `@mariozechner/pi-ai`) + 独立工具权限网关 |
| **语音合成** | Streaming TTS | Qwen3-TTS (`Qwen/Qwen3-TTS-12Hz-0.6B`) + 句级流式缓冲 |
| **推理与训练后端** | Local Backend / R&D | Python 3.11+ / PyTorch + CUDA / FastAPI + Local WebSocket |
| **测试与评测** | Testing & Benchmark | Vitest (TS) + pytest (Python) + 自定义端到端评测套件 |

完整选型依据、版本基线与 8 GB 显存分配预算请参阅 **[技术栈选型与架构规范](docs/TECH_STACK.md)**。

---

## 仓库目录结构

```text
VoiceAgent/
├── apps/
│   └── desktop/             # Electron 桌面端应用 (main / preload / renderer)
├── packages/
│   ├── audio/               # 麦克风采集、16kHz 重采样、环形/句首缓冲、播放队列与进度追踪
│   ├── speech/              # VAD、流式 ASR/TTS 客户端封装与句级文本切分净化
│   ├── interaction/         # 混合轮次管理、打断控制、播放状态同步与推测式生成
│   ├── agent/               # Pi SDK 封装、三级权限控制、授权工作流与本地工具集
│   ├── protocol/            # Local WebSocket 消息信封、二进制帧头与共享 Schema
│   └── metrics/             # 端到端分阶段延迟、RTF、队列水位与显存监控采集
├── inference/               # 本地 Python 语音推理服务 (ASR / Turn Adapter / TTS / WebSocket Server)
├── training/                # 离线训练与微调流水线 (ASR / Turn Adapter / TTS)
├── evaluation/              # 模块级指标与 10 类系统级交互场景评测套件
├── configs/                 # 运行时与实验 YAML 配置文件 (default.yaml)
├── data/                    # 本地数据集、会话划分清单与评测集目录 (Git 忽略音频数据)
├── models/                  # 本地模型缓存与 Checkpoint 注册表目录 (Git 忽略权重文件)
├── tests/                   # 跨模块集成测试与回归测试
└── docs/                    # 项目文档中心
    ├── live/
    │   └── VoiceAgent_PRD.md # 权威产品需求文档 (PRD V1.0)
    ├── README.md             # 文档导航索引
    ├── TECH_STACK.md         # 技术栈选型、显存与延迟预算规范
    ├── MODULES.md            # 全模块职责边界与代码接口设计说明书
    ├── PROTOCOL.md           # 本地 WebSocket 实时音频与事件通信协议规范
    └── DEVELOPMENT.md        # 开发环境搭建、工程规范与分阶段实施指南
```

---

## 文档导航

| 文档 | 内容说明 |
|---|---|
| **[产品需求文档 (PRD V1.0)](docs/live/VoiceAgent_PRD.md)** | 产品定位、参考项目融合策略（`local-voice` + `Voice-Light`）、FR-01~FR-36 功能需求、性能指标与实验体系 |
| **[文档中心索引](docs/README.md)** | 按研发角色（前端/系统工程、语音算法、评测验证）组织的阅读路径 |
| **[技术栈与架构规范](docs/TECH_STACK.md)** | 分层技术选型、双进程架构决策、8 GB GPU 显存预算、端到端延迟拆解与降级策略 |
| **[模块功能与接口设计](docs/MODULES.md)** | 9 大核心子系统的职责划分、状态机流转、TypeScript / Python 核心接口定义与异常边界 |
| **[实时通信协议规范](docs/PROTOCOL.md)** | `127.0.0.1` WebSocket 二进制音频帧格式、三级会话 ID 隔离、15+ 核心事件定义与全双工时序图 |
| **[开发与研发路线指南](docs/DEVELOPMENT.md)** | Windows 11 + CUDA 环境配置、Monorepo 命令、安全红线、实验记录规范与 Phase 0–5 验收标准 |

---

## 开发路线（Roadmap）

- [ ] **Phase 0：技术验证** — 验证 Windows / WSL2 CUDA 环境、Nemotron 流式推理与中间层特征提取、Qwen3-TTS 流式合成及 8 GB 显存共存基准。
- [ ] **Phase 1：基础语音 Agent** — 打通 Electron 桌面端、音频采集播放、本地 ASR/TTS 服务、Pi SDK 多轮对话与受权限控制的工具调用（≥20 轮连续稳定对话）。
- [ ] **Phase 2：全双工语音交互** — 实现 AEC 回声消除、Duck/Pause/Resume/Cancel 打断控制、播放进度追踪与混合轮次决策。
- [ ] **Phase 3：Turn Adapter 研发** — 复现 Voice-Light 因果 GRU Adapter，完成合成预训练、授权对话微调、基线对比与消融实验。
- [ ] **Phase 4：ASR / TTS 微调** — 完成 Bad Case 归因、ASR 口音/指令微调、TTS 零样本基线对比与说话人音色微调部署。
- [ ] **Phase 5：性能优化与发布** — 推进推测式生成（Speculative Generation）、精度量化与显存调度优化，完成 ≥4 小时稳定性压测与实验报告交付。

---

## 许可证与合规声明

本项目仓库源码采用 [MIT License](LICENSE) 开源。项目所引用的第三方预训练模型（如 NVIDIA Nemotron、Qwen3-TTS、Silero VAD）、外部 SDK（如 Pi Agent）及公开研究数据集各自遵循其原始许可证条款，部署与分发前请核查对应授权要求。
