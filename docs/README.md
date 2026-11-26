# VoiceAgent 文档中心（Documentation Index）

本目录汇总了 **VoiceAgent** 的产品需求基线、技术架构决策、模块级接口说明书、实时通信协议以及工程与算法研发指南。

---

## 1. 文档目录结构

```text
docs/
├── live/
│   └── VoiceAgent_PRD.md    # [权威需求基线] 产品需求文档 V1.0
├── README.md                # [当前文件] 文档总览与阅读导航
├── TECH_STACK.md            # 技术栈选型、跨语言架构、显存预算与延迟预算规范
├── MODULES.md               # 全系统模块功能拆解、状态机与 TypeScript/Python 接口契约
├── PROTOCOL.md              # Local WebSocket 二进制音频帧与控制事件通信协议规范
└── DEVELOPMENT.md           # 开发环境配置、工程与实验规范、Phase 0–5 实施与验收手册
```

> **关于 `docs/live/` 的约定**：`docs/live/` 存放当前生效的权威产品需求与活文档基线（Living Specifications）。任何涉及功能边界、优先级（P0–P3）或验收指标的变更，均应先同步更新 `docs/live/VoiceAgent_PRD.md`，再推进对应模块设计变更。

---

## 2. 按场景快速阅读指南

### 初次了解项目
1. 阅读 [VoiceAgent_PRD.md](live/VoiceAgent_PRD.md) 第 1–3 章，理解产品定位与 `local-voice` + `Voice-Light` 的融合架构。
2. 阅读 [TECH_STACK.md](TECH_STACK.md)，明确为什么采用 **TypeScript 桌面编排层 + Python 本地推理/训练后端** 的双层架构，以及 8 GB 显存下的资源分配策略。

### 开发桌面端、音频链路与 Agent 编排（TypeScript）
1. 阅读 [MODULES.md](MODULES.md) 第 1–6 章，掌握 `apps/desktop` 与 `packages/{audio,speech,interaction,agent,protocol,metrics}` 的职责边界、核心接口与状态机。
2. 对照 [PROTOCOL.md](PROTOCOL.md) 实现 WebSocket 客户端、`session_id` / `turn_id` / `generation_id` 事件过滤以及播放进度确认（`playback.ack`）。

### 开发本地语音推理、模型微调与轮次预测算法（Python）
1. 阅读 [MODULES.md](MODULES.md) 第 7–9 章，明确 `inference/`、`training/` 与 `evaluation/` 的模块契约、Nemotron Encoder 特征提取点以及 Causal GRU Turn Adapter 的因果状态约束。
2. 阅读 [DEVELOPMENT.md](DEVELOPMENT.md) 中的训练数据防泄漏规范、Checkpoint 注册流程与标准化实验记录要求（对照实验 `ASR-A0~A3`、`TURN-A0~A5`、`TTS-A0~A3`）。

---

## 3. 核心设计准则速查

1. **在线推理与离线训练严格解耦**：任何离线训练失败或实验模型指标不达标，均不得影响桌面端加载已有稳定 Checkpoint 正常运行。
2. **全双工双轨状态独立维护**：用户监听轨（`Listening`）与助手输出轨（`Thinking` / `Executing` / `Speaking`）并行存在，打断控制严格区分**音频播放取消（Cancel）**与**Agent 生成取消（Abort）**，已执行的外部工具副作用绝不因打断而隐式回滚。
3. **TTS 输入严格净化**：代码片段、工具调用 JSON 参数、模型内部推理块与原始工具返回数据严禁直接流入 TTS 队列，仅允许面向用户的自然语言回复进入句级合成缓冲。
4. **安全与隐私默认最小化**：本地推理网关仅监听 `127.0.0.1`；默认不持久化麦克风原始录音；工作目录不等于安全沙箱，所有具有外部副作用的工具调用必须经过独立权限网关校验。
