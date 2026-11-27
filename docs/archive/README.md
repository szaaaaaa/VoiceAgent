# VoiceAgent 历史归档文档（Archive）

本目录保存 VoiceAgent 在架构演进过程中已退役的历史需求与技术设计文档，仅供版本溯源与设计对比参考，**不再作为当前开发与实验的执行依据**。

当前生效的权威 PRD 与技术规范请查阅 **[`docs/live/`](../live/)**，实测实验记录请查阅 **[`docs/experiments/`](../experiments/)**。

---

## `v1/` — V1.0 独立桌面语音助手架构（已归档）

### 归档原因（V1 → V2 架构收敛）
1. **核心重心聚焦语音算法研发**：项目核心壁垒在于 **流式 ASR 微调、流式 TTS 音色微调、冻结 Encoder 特征驱动的 Causal GRU 轮次预测/打断算法及 8 GB 消费级显卡推理优化**，而非重复开发一套独立的桌面 Agent 壳层。
2. **无缝热插拔接入既有二次开发 Pi Agent**：本机已有成熟二次开发的 Pi Agent 运行环境，语音交互能力调整为标准 **Pi Package 扩展模块（`packages/pi-voice`，提供 `/voice` 命令与快捷键）**，像 AGY / Codex 语音模式一样热插拔挂载，零侵入复用既有 Agent 会话、工具链与配置。
3. **桌面端转型为算法实验与观测工作台**：原 Electron 桌面端（`apps/desktop`）从“独立聊天外壳”重新定位为 **语音算法实验与实时观测工作台（R&D Workbench）**，专职负责 VAD / Causal GRU 三头概率示波、Checkpoint 热切换与 A/B 盲测对比、延迟/显存剖析及 Bad Case 音频回放。

### V1 归档文件与 V2 活文档映射表

| V1 归档文件 (`docs/archive/v1/`) | V2 现行权威文件 (`docs/live/`) | 主要演进内容 |
|---|---|---|
| [VoiceAgent_PRD_v1.md](v1/VoiceAgent_PRD_v1.md) | [VoiceAgent_PRD.md](../live/VoiceAgent_PRD.md) | 重构为「Python 语音训练推理内核 + Pi `/voice` 热插拔扩展 + 桌面实验观测工作台」三层架构；纳入云盘/流式数据集约束与 6 类全双工轮次场景 |
| [TECH_STACK_v1.md](v1/TECH_STACK_v1.md) | [TECH_STACK.md](../live/TECH_STACK.md) | 新增 `@earendil-works/pi-coding-agent` 扩展宿主规范、Python WASAPI 原生音频守护模式、RTX 5070 8GB 四模型实测显存基线与 TTS 加速路线 |
| [MODULES_v1.md](v1/MODULES_v1.md) | [MODULES.md](../live/MODULES.md) | 新增 `packages/pi-voice` 插件模块设计，重构 `apps/desktop` 为实验观测工作台，细化 `training/` 云端流式数据与 `CheckpointRegistry` 接口 |
| [PROTOCOL_v1.md](v1/PROTOCOL_v1.md) | [PROTOCOL.md](../live/PROTOCOL.md) | 扩展统一本地控制面协议：支持双音频源模式（`local_device` / `ws_pcm`）、多客户端观测广播（Pi 插件运行 + 桌面工作台旁路示波）与 A/B 实验接口 |
| [DEVELOPMENT_v1.md](v1/DEVELOPMENT_v1.md) | [DEVELOPMENT_AND_EXPERIMENTS.md](../live/DEVELOPMENT_AND_EXPERIMENTS.md) | 整合 Pi 插件本地热挂载流程、Google Drive / HF `streaming=True` 零本地囤积规范、AutoDL ¥100 预算微调清单与阶段验收门禁 |
