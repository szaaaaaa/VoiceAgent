# VoiceAgent 产品需求文档（PRD）

**版本：** V1.0  
**产品名称：** VoiceAgent（暂定）  
**产品类型：** 本地全双工语音智能体  
**目标平台：** Windows 11 + NVIDIA GPU  
**开发语言：** TypeScript + Python  
**参考项目：** local-voice、Voice-Light

---

## 1. 产品概述

### 1.1 产品定位

VoiceAgent 是一个基于 Pi Agent 的桌面级实时语音助手，支持本地语音识别、语音合成、全双工语音交互、工具调用及计算机操控。

产品采用 ASR–LLM–TTS 级联架构。语音模型在本地运行，LLM 通过 Pi Agent 接入，允许用户通过自然语音完成信息查询、文件处理和计算机操作等任务。

同时，VoiceAgent 是一个语音算法研发与评测平台，支持语音模型的独立训练、微调、替换、部署和性能评估。

### 1.2 产品目标

- 实现低延迟、连续、可打断的自然语音交互。
- 支持 Pi Agent 的模型配置、会话管理和工具调用。
- 实现本地 ASR、TTS 模型部署，避免依赖商业语音 API。
- 建立可训练的语音轮次预测和智能打断检测模块。
- 支持 ASR、TTS 模型微调及效果评测。
- 在消费级 NVIDIA GPU 上实现稳定运行。

### 1.3 目标用户

- 希望通过语音操控个人电脑的用户。
- 需要本地化语音交互能力的 Agent 开发者。
- 需要训练、测试及部署语音模型的算法工程师。

---

## 2. 参考项目与融合策略

### 2.1 local-voice

GitHub：https://github.com/mthomas100/local-voice

项目定位：面向 Apple Silicon 的本地实时语音助手。

原始架构：

```text
Mac / iPhone / Browser
          ↓
Pipecat Orchestrator
          ↓
Silero VAD + Smart Turn
          ↓
Nemotron / Parakeet ASR
          ↓
Echo Guard
          ↓
Space Router
          ↓
Pi Agent RPC
          ↓
LLM + Tools
          ↓
Sentence Splitter
          ↓
Qwen3-TTS / Kokoro
          ↓
Audio Playback
```

主要参考内容：

- Pi Agent 与语音系统的集成方式。
- 可替换的语音模型 Adapter。
- 流式语音处理与句级 TTS 调度。
- Barge-in（用户语音打断）。
- 音频播放进度追踪。
- Spaces（工作空间）管理。
- 工具调用权限控制与语音授权。
- 端到端延迟监控。

不直接迁移的部分：

- Apple MLX 推理后端。
- macOS、iOS 原生客户端。
- 后台 Brain 和日记系统。
- Apple 平台专用 GPU 调度机制。

### 2.2 Voice-Light

GitHub：https://github.com/BertilBraun/Voice-Light

项目定位：面向自然全双工语音交互的模型训练、算法研究与部署系统。

原始架构：

```text
Browser Microphone
          ↓
WebSocket / VoiceSession
          ↓
Silero VAD
          ↓
Streaming Nemotron ASR
     ├── ASR Decoder
     └── Encoder Features
               ↓
         Causal Turn Adapter
               ↓
        Dialogue Controller
               ↓
         Qwen LLM + Tools
               ↓
         Kyutai Streaming TTS
               ↓
         Browser Playback
```

主要参考内容：

- 基于 Nemotron Encoder 中间特征的轮次预测模型。
- 因果 GRU Adapter 训练。
- 合成对话数据生成与标注。
- 双通道人类对话数据处理。
- 发言完成、停顿、附和与主动打断检测。
- 可恢复的音频暂停与音量控制。
- Speculative Generation（推测式生成）。
- 基于实际播放进度的会话历史管理。
- 可复现的训练与评测流程。

Voice-Light 的公开实验中，训练得到的轮次完成预测模型未超过较强的传统时序基线。因此，本项目保留混合轮次控制策略，将 Adapter 的实际收益作为待验证的算法问题。

### 2.3 融合策略

| 模块 | local-voice | Voice-Light | VoiceAgent |
|---|---|---|---|
| 应用层 | Python / Swift | Python / Web | TypeScript |
| GUI | Mac / iPhone | Web | Electron + React |
| ASR | Nemotron / Parakeet | Nemotron | Nemotron |
| VAD | Silero | Silero | Silero |
| 轮次预测 | Smart Turn | Causal Turn Adapter | 混合轮次预测 |
| LLM | Pi Agent RPC | Qwen | Pi Agent SDK |
| TTS | Qwen3-TTS / Kokoro | Kyutai | Qwen3-TTS |
| 打断控制 | Pause / Interrupt | Duck / Pause / Resume | 混合打断控制 |
| 模型训练 | 非核心目标 | Adapter、LLM | ASR、TTS、Adapter |
| 部署平台 | Apple Silicon | 云端 GPU | Windows / NVIDIA |
| Computer Use | 非核心功能 | 非核心功能 | Pi Tool |

以 local-voice 的产品架构为基础，融合 Voice-Light 的语音交互算法及训练体系，并复用原实时翻译项目中的音频处理、缓冲队列和延迟监控设计。

---

## 3. 系统总体架构

### 3.1 在线推理架构

```text
┌──────────────────────────────────────────┐
│           Electron + React GUI           │
│   Transcript / Status / Tools / Metrics  │
└─────────────────────┬────────────────────┘
                      │
                Microphone
                      ↓
         Audio Capture / Resampling
                      ↓
          AEC / Noise Suppression
                      ↓
                  Silero VAD
                      ↓
          Nemotron Streaming Encoder
                ┌─────┴──────┐
                ↓            ↓
          ASR Decoder    Turn Adapter
                │            │
                └─────┬──────┘
                      ↓
                Turn Manager
                      ↓
                Pi Agent SDK
                ┌─────┴──────┐
                ↓            ↓
               LLM       Agent Tools
                │            │
                └─────┬──────┘
                      ↓
                Text Streaming
                      ↓
                Sentence Buffer
                      ↓
              Streaming Qwen3-TTS
                      ↓
                Audio Playback
                      │
                      └────→ AEC Reference
```

Turn Manager 同时控制 Agent 生成、TTS 合成和音频播放的暂停、恢复与取消。

### 3.2 离线训练架构

```text
Public Datasets / Authorized Recordings
                    ↓
          Data Processing Pipeline
                    ↓
        Train / Validation / Test Split
                    ↓
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
 ASR Training  Turn Training  TTS Training
      ↓             ↓             ↓
 ASR Weights    Turn Adapter   TTS Weights
      └─────────────┼─────────────┘
                    ↓
           Model Evaluation
                    ↓
           Checkpoint Registry
                    ↓
          Local Inference Backend
```

在线推理与离线训练相互独立。训练失败或模型效果不达标时，不影响桌面应用使用已有模型。

---

## 4. 产品功能需求

优先级定义：

| 优先级 | 定义 |
|---|---|
| P0 | 基础产品必须实现 |
| P1 | 核心语音交互与算法研发 |
| P2 | 模型优化与评测扩展 |
| P3 | 后续产品扩展 |

### 4.1 音频采集与处理

| 编号 | 功能 | 优先级 |
|---|---|---|
| FR-01 | 麦克风实时音频采集 | P0 |
| FR-02 | 音频输入、输出设备选择 | P0 |
| FR-03 | 音频重采样与单声道转换 | P0 |
| FR-04 | Ring Buffer 与 Pre-roll Buffer | P0 |
| FR-05 | 回声消除（AEC） | P1 |
| FR-06 | 背景噪声抑制 | P1 |
| FR-07 | 系统内录与音频文件输入 | P2 |

技术要求：

- ASR 输入统一转换为 16 kHz 单声道 PCM。
- 音频帧携带时间戳。
- 音频输入保持连续，避免 VAD 切分破坏上下文。
- 保留句首缓存，防止语音起始部分丢失。
- AEC 接收实际播放音频作为回声参考信号，或使用经验证的系统级回声消除能力。

### 4.2 实时语音识别（ASR）

| 编号 | 功能 | 优先级 |
|---|---|---|
| FR-08 | 本地流式 ASR | P0 |
| FR-09 | Partial Transcript | P0 |
| FR-10 | Final Transcript | P0 |
| FR-11 | ASR 状态缓存管理 | P0 |
| FR-12 | Encoder 中间特征提取 | P1 |
| FR-13 | 热词与上下文增强 | P2 |
| FR-14 | 微调模型加载与切换 | P1 |

默认研究模型：

`nvidia/nemotron-speech-streaming-en-0.6b`

首版以英文语音为主，保持与 Voice-Light 的 ASR Backbone 和 Adapter 设计一致。

后续支持中文及多语言 ASR。更换 Backbone 时，需要重新验证 Encoder 特征接口及 Adapter 兼容性。

### 4.3 语音活动检测（VAD）

采用 Silero VAD。

负责：

- 检测语音开始与结束。
- 区分语音和静音。
- 触发轮次预测。
- 检测助手播放期间的潜在用户发言。
- 为 ASR 提供端点和音频边界信息。

VAD 不独立决定用户是否完成整轮发言。

### 4.4 对话轮次管理（Turn Manager）

| 编号 | 功能 | 优先级 |
|---|---|---|
| FR-15 | 发言结束判断 | P0 |
| FR-16 | 句中停顿识别 | P1 |
| FR-17 | 用户主动打断识别 | P1 |
| FR-18 | 用户附和识别 | P1 |
| FR-19 | 轮次预测模型切换 | P1 |
| FR-20 | 保守超时回退 | P1 |

系统需要区分以下交互事件：

| 事件 | 示例 | 系统行为 |
|---|---|---|
| EOT：发言完成 | “帮我打开浏览器。” | 提交给 Agent |
| HOLD：句中停顿 | “帮我打开……嗯……浏览器。” | 继续监听 |
| Backchannel：附和 | “嗯”“OK” | 保留当前回复，必要时恢复播放 |
| Interruption：主动打断 | “等一下，我改个要求。” | 停止当前回复，接收新指令 |
| Noise：环境噪声 | 咳嗽、键盘声 | 不取消当前任务 |

采用混合控制策略：

```text
VAD Evidence
      +
Turn Adapter Probability
      +
Assistant Playback State
      +
ASR Transcript
      +
Silence Timeout
      ↓
Turn Decision
```

Smart Turn 和固定静音阈值作为对照基线与可用回退方案。

### 4.5 全双工交互

全双工模式下，助手播放语音时，麦克风仍持续采集和处理用户声音。

处理流程：

```text
Assistant Speaking
        ↓
Potential User Speech
        ↓
Reduce Volume / Pause Playback
        ↓
Collect Audio + Predict Interaction
        ↓
 ┌────────────┬──────────────┐
 ↓            ↓              ↓
Noise     Backchannel    Interruption
 ↓            ↓              ↓
Resume       Resume     Cancel Reply
                              ↓
                      Process New Turn
```

系统必须支持：

- Duck：暂时降低播放音量。
- Pause：暂停播放并保留当前位置。
- Resume：恢复原有语音。
- Cancel：清除当前回复的待播放音频。
- Abort：请求取消对应的 Agent 生成任务。

语音播放取消与 Agent 工具取消分别管理。已经执行的外部操作不能因用户打断而自动撤销。

### 4.6 Pi Agent 集成

| 编号 | 功能 | 优先级 |
|---|---|---|
| FR-21 | Pi SDK 接入 | P0 |
| FR-22 | 复用 Pi 模型配置 | P0 |
| FR-23 | 多轮会话管理 | P0 |
| FR-24 | 增量文本事件 | P0 |
| FR-25 | 工具调用与状态展示 | P0 |
| FR-26 | Agent 生成取消 | P0 |
| FR-27 | 工具权限控制 | P0 |
| FR-28 | 工作空间切换 | P2 |
| FR-29 | Computer Use | P3 |

Pi Agent 负责：

- LLM 接入。
- 上下文管理。
- Session 管理。
- Skills 和 Tools。
- 任务推理与执行。

VoiceAgent 不重新实现 Agent Runtime。

### 4.7 实时语音合成（TTS）

默认候选模型：Qwen3-TTS 0.6B。

| 编号 | 功能 | 优先级 |
|---|---|---|
| FR-30 | 本地语音合成 | P0 |
| FR-31 | 流式音频输出 | P1 |
| FR-32 | 句级文本缓冲 | P0 |
| FR-33 | 音频播放队列 | P0 |
| FR-34 | 播放取消与恢复 | P1 |
| FR-35 | 自定义音色模型 | P2 |
| FR-36 | TTS 微调模型加载 | P1 |

TTS 仅接收适合朗读的自然语言文本。

工具调用参数、代码、内部推理信息和原始工具输出不得直接进入 TTS 播放队列。

### 4.8 推测式生成（Speculative Generation）

优先级：P2。

当轮次预测模型判断用户即将结束发言时，系统可以提前启动候选回复生成。

```text
Partial Transcript
        ↓
Speculative EOT Prediction
        ↓
Start Candidate Generation
        ↓
Confirm Final Transcript
     ┌──┴──┐
     ↓     ↓
  Valid  Invalid
     ↓     ↓
 Commit  Discard
```

候选回复在确认前不得播放、写入正式会话历史或执行外部工具。

若最终识别结果发生实质性变化，取消候选回复并重新生成。

### 4.9 工具调用与权限管理

参考 local-voice 的 Spaces 和 Spoken Approvals 机制。

支持两种运行模式：

| 模式 | 权限 |
|---|---|
| Conversation | 普通对话及允许的只读工具 |
| Action | 执行经过授权的本地操作 |

支持三种权限级别：

- Readonly：禁止修改外部状态。
- Ask：执行有副作用的操作前请求用户授权。
- Trusted：仅允许用户明确配置的有限操作自动执行。

文件删除、命令执行、数据发送及其他高风险操作默认需要明确确认。

授权界面展示实际工具、目标路径、命令、修改内容和预计影响。沉默、超时及含义不明确的语音均视为未授权。

工作目录本身不构成安全沙箱，文件访问范围及命令权限必须由工具层独立检查。

---

## 5. 用户界面需求

### 5.1 主界面

```text
┌─────────────────────────────────────────────┐
│ VoiceAgent    Model / Session / Settings    │
├─────────────────────────────────────────────┤
│                                             │
│ User: Help me find the latest GPU drivers.  │
│                                             │
│ Agent: Let me check that for you.           │
│                                             │
│ Tool: Searching...                          │
│                                             │
├─────────────────────────────────────────────┤
│              Audio Waveform                 │
│                Listening                    │
│                                             │
│          [ Mic ] [ Stop ] [ Mute ]          │
├─────────────────────────────────────────────┤
│ ASR  | LLM  | TTS  | Latency  | GPU Memory  │
└─────────────────────────────────────────────┘
```

### 5.2 交互模式

- Hands-free：持续监听，自动判断用户发言结束。
- Push-to-talk：按键开始和结束发言。
- Text：文本输入。
- Mute：关闭麦克风采集。

### 5.3 状态展示

监听状态与回复状态独立维护，允许同时处于 Listening 和 Speaking。

| 状态 | 含义 |
|---|---|
| Idle | 未进行交互 |
| Listening | 正在采集音频 |
| Thinking | Agent 正在生成回复 |
| Executing | Agent 正在调用工具 |
| Speaking | 正在播放语音 |
| Interrupted | 当前回复被打断 |
| Awaiting Approval | 等待用户授权 |
| Error | 模块运行异常 |

### 5.4 开发者模式

展示：

- 实时音频波形。
- VAD 概率。
- ASR Partial / Final Transcript。
- Turn Adapter 输出概率。
- LLM 首 Token 延迟。
- TTS 首包及首音频播放延迟。
- GPU 显存与利用率。
- 队列长度。
- 用户打断与播放状态事件。
- 当前模型版本及配置。

支持切换模型、导出实验日志和回放测试音频。

---

## 6. 算法研发需求

### 6.1 ASR 微调

目标：改善基础模型在目标口音、语音指令及复杂环境中的识别表现。

训练流程：

1. 基线模型评测及 Bad Case 分析。
2. 确定具体识别缺陷。
3. 构建或筛选训练语料。
4. 完成音频清洗、切分、标注及数据增强。
5. 进行全参数或参数高效微调。
6. 验证流式推理兼容性。
7. 将训练后的模型接入系统。

对照实验：

| 实验 | 方法 |
|---|---|
| ASR-A0 | 原始模型 |
| ASR-A1 | 上下文或热词增强 |
| ASR-A2 | 领域或口音微调 |
| ASR-A3 | 微调 + 数据增强 |

评估指标：WER、CER、指令识别准确率、RTF、推理显存。

保留通用语音测试集，检查微调后的通用能力退化情况。

### 6.2 Turn Adapter 训练

参考 Voice-Light 的冻结 ASR Encoder 与因果 GRU Adapter。

```text
Frozen Nemotron Encoder
           ↓
    Multi-layer Features
           +
    Assistant Playback State
           ↓
      Causal GRU Adapter
           ↓
     Classification Heads
     ├── Completion
     ├── Floor Taking
     └── Non-floor Feedback
```

训练要求：

- 首先冻结 ASR Encoder，训练独立 Adapter。
- 仅使用当前及历史可用特征，避免未来信息泄漏。
- 保留跨音频块的模型状态。
- 支持合成对话数据预训练。
- 在获得授权的人类对话数据后进行适配。
- 按完整会话划分训练、验证和测试集。
- 在验证集确定阈值，测试集不得用于调参。

数据来源：

- Voice-Light Synthetic Audio：公开合成语音与对话重建数据。
- Voice-Light Human Conversation Corpus：访问受限。
- Mundo TurnBench Dev：公开人工轮次标注数据。
- 自行采集并标注的授权对话数据。

实验组：

| 实验 | 方法 |
|---|---|
| TURN-A0 | 固定静音阈值 |
| TURN-A1 | Silero VAD |
| TURN-A2 | Smart Turn |
| TURN-A3 | Voice-Light Adapter |
| TURN-A4 | 自训练 Adapter |
| TURN-A5 | Adapter + 混合交互控制 |

发言结束检测与主动打断检测分别评估。

评价指标包括 False Cutoff Rate、EOT Recall、F1、Backchannel False-cancel Rate 和 Decision Latency。

### 6.3 TTS 微调

目标：实现目标说话人的音色适配，并研究微调对发音质量、自然度和生成稳定性的影响。

数据来源为本人录音、获得授权的录音或符合使用协议的公开数据集。

对照实验：

| 实验 | 方法 |
|---|---|
| TTS-A0 | 原始预训练模型 |
| TTS-A1 | Zero-shot Voice Cloning |
| TTS-A2 | 少量数据微调 |
| TTS-A3 | 不同训练数据量微调 |

评估指标：

- MOS：人工主观自然度评价。
- Speaker Similarity：音色相似度。
- CER / WER：合成语音内容准确性。
- TTFA：首音频生成时间。
- RTF：实时率。

零样本声音克隆作为必要基线，用于验证微调是否具有实际收益。

### 6.4 推理优化

研究内容：

- ASR Chunk Size 与 Lookahead。
- FP16、BF16、INT8 等支持的推理精度。
- ASR 和 TTS 并发执行。
- GPU 显存分配。
- 模型常驻与加载策略。
- TTS 文本缓冲策略。
- 音频队列与任务取消。
- 推测式生成的延迟收益与无效计算成本。

---

## 7. 技术栈与项目结构

### 7.1 技术栈

| 模块 | 技术选型 |
|---|---|
| Desktop | Electron |
| Frontend | React + TypeScript |
| Application Runtime | Node.js |
| Audio Capture | Web Audio API / AudioWorklet |
| Audio Processing | WebRTC Audio Processing 或经验证的系统音频方案 |
| VAD | Silero VAD |
| ASR | Nemotron Streaming |
| TTS | Qwen3-TTS |
| Turn Adapter | PyTorch + GRU |
| Agent Runtime | Pi SDK |
| Inference | PyTorch / CUDA，后续评估 ONNX |
| Training | Python + PyTorch |
| Communication | Local WebSocket |
| Evaluation | Python |
| Testing | Vitest + pytest |

应用层采用 TypeScript，模型推理与训练层采用 Python。

暂不强制将所有语音模型转换为 ONNX，以保留 Encoder 中间特征访问及模型修改能力。

### 7.2 目录结构

```text
VoiceAgent/
├── apps/
│   └── desktop/
│       ├── main/
│       ├── preload/
│       └── renderer/
│
├── packages/
│   ├── audio/
│   │   ├── capture.ts
│   │   ├── playback.ts
│   │   ├── ring-buffer.ts
│   │   └── resampler.ts
│   │
│   ├── speech/
│   │   ├── asr-client.ts
│   │   ├── tts-client.ts
│   │   └── vad.ts
│   │
│   ├── interaction/
│   │   ├── turn-manager.ts
│   │   ├── interruption.ts
│   │   ├── playback-state.ts
│   │   └── speculative.ts
│   │
│   ├── agent/
│   │   ├── pi.ts
│   │   ├── permissions.ts
│   │   └── tools/
│   │
│   ├── protocol/
│   └── metrics/
│
├── inference/
│   ├── asr/
│   ├── tts/
│   ├── turn_adapter/
│   └── server.py
│
├── training/
│   ├── asr/
│   ├── tts/
│   └── turn_adapter/
│
├── evaluation/
│   ├── asr/
│   ├── tts/
│   ├── turn_taking/
│   ├── interruption/
│   └── end_to_end/
│
├── data/
├── models/
├── configs/
├── tests/
├── docs/
├── README.md
└── LICENSE
```

### 7.3 模块通信协议

参考 local-voice 的 WebSocket Protocol，定义以下核心事件。

| 事件 | 功能 |
|---|---|
| `audio.frame` | 输入音频帧 |
| `vad.start` | 检测到语音 |
| `vad.end` | 检测到静音 |
| `asr.partial` | 增量识别文本 |
| `asr.final` | 最终识别文本 |
| `turn.prediction` | 轮次预测结果 |
| `turn.commit` | 提交用户发言 |
| `agent.delta` | Agent 增量文本 |
| `tool.start` / `tool.end` | 工具执行状态 |
| `approval.request` | 请求授权 |
| `approval.response` | 用户授权结果 |
| `tts.chunk` | 生成的音频块 |
| `playback.ack` | 实际播放进度 |
| `generation.cancel` | 取消当前生成任务 |
| `error` | 模块异常 |

每轮交互使用 Session ID、Turn ID 和 Generation ID 关联事件，避免被取消的旧任务输出进入新一轮对话。

### 7.4 会话历史管理

分别维护：

- 用户最终转写文本。
- Agent 正式回复。
- 实际播放过的回复内容。
- 工具调用及其执行结果。
- 未提交的候选生成内容。

被打断后，下一轮对话应让 Agent 知道用户实际听到了哪些内容。

已执行工具及结果必须保留在任务上下文中，不能因语音未播放而被丢弃。

---

## 8. 非功能性需求

### 8.1 性能指标

以下为拟定研发目标，尚未经过目标硬件实测。

| 指标 | V1 目标 | 优化目标 |
|---|---:|---:|
| 发言结束至首音频播放，P50 | ≤2 s | ≤1 s |
| 发言结束至首音频播放，P95 | ≤4 s | ≤2 s |
| 潜在打断至播放暂停，P95 | ≤250 ms | ≤150 ms |
| 确认打断至停止旧回复，P95 | ≤500 ms | ≤300 ms |
| ASR / TTS RTF | ＜1 | 持续降低 |
| GPU 显存 | ≤8 GB | 保留运行余量 |
| 连续运行 | ≥1 h | ≥4 h |

延迟测试需要分别报告：

- 冷启动与热启动。
- 普通对话与工具调用任务。
- 单模型与并发模型。
- 各模块计算时间与排队时间。

端到端延迟以用户实际语音结束至第一段可听回复播放为准，不能用服务器生成首个 PCM 数据包代替。

### 8.2 稳定性

- 支持模型服务异常检测和重启。
- 支持推理请求超时及取消。
- 防止音频缓冲区无限增长。
- 防止已取消任务的音频继续播放。
- 模型不可用时允许回退至文本交互。
- 资源不足时提示用户切换轻量模型。

### 8.3 隐私与安全

- 语音模型默认本地运行。
- 推理服务默认仅监听本机地址。
- 默认不长期保存原始麦克风录音。
- 开启训练数据采集时，需要明确提示。
- 使用在线 LLM 时，转写文本及必要上下文会发送给对应模型服务。
- API Key 不进入代码仓库及实验日志。
- 开源代码、模型及数据分别遵守其许可证。
- 高风险工具调用必须经过独立权限校验。

### 8.4 可扩展性

ASR、TTS、VAD、Turn Detector 与 Agent Connector 均通过明确的模块接口接入。

更换模型不应要求重写 GUI 或对话控制器；更换依赖 Encoder 特征的模型时，应重新验证对应 Adapter。

---

## 9. 实验与评测体系

### 9.1 模块级评测

| 模块 | 指标 |
|---|---|
| ASR | WER、CER、术语准确率、RTF |
| TTS | MOS、音色相似度、CER、TTFA |
| VAD | Precision、Recall、F1 |
| Turn Adapter | F1、False Cutoff Rate、EOT Recall |
| Interruption | 误打断率、漏检率、停止延迟 |
| AEC | ERLE、回声导致的 ASR 误识别率 |
| Agent | 工具调用成功率、错误执行率 |
| System | P50/P95 延迟、显存、吞吐量 |

### 9.2 系统级测试场景

| 场景 | 测试目标 |
|---|---|
| 正常连续对话 | 基础交互能力 |
| 长时间句中停顿 | 是否错误结束发言 |
| 用户主动打断 | 是否及时停止旧回复 |
| 用户短暂附和 | 是否错误取消回复 |
| 咳嗽及背景噪声 | 误触发控制 |
| 扬声器播放 | 是否将助手语音识别成用户指令 |
| 工具执行期间打断 | 任务状态与取消行为 |
| 未授权工具调用 | 是否被正确拦截 |
| GPU 资源不足 | 降级与异常恢复 |
| 长时间连续运行 | 延迟漂移与内存泄漏 |

### 9.3 实验结果管理

每次实验保存：

- Experiment ID。
- Git Commit。
- 数据集版本与数据划分。
- 模型及 Checkpoint 版本。
- 超参数。
- 硬件及运行环境。
- 原始指标。
- Bad Cases。
- 实验结论。

允许保留负面实验结果。未在独立测试集证明有效的优化不得作为性能提升成果。

---

## 10. 开发路线与验收标准

### Phase 0：技术验证

任务：

- 固定参考仓库版本。
- 验证 Windows / WSL2 CUDA 环境。
- 运行 Nemotron 和 Qwen3-TTS。
- 验证 Encoder 特征提取。
- 测试 8GB 显存下的模型共存能力。
- 检查开源代码及模型授权。

验收标准：确定可在目标设备运行的 ASR、TTS 和 Adapter 技术组合。

### Phase 1：基础语音 Agent

任务：

- Electron 桌面应用。
- TypeScript 音频采集及播放。
- 本地 ASR 和 TTS 推理服务。
- Pi SDK 接入。
- 基础会话及工具调用。

验收标准：连续完成至少 20 轮语音对话，支持文本输入及受权限控制的工具调用。

### Phase 2：全双工语音交互

任务：

- 实时流式 ASR / TTS。
- 回声消除。
- 语音打断。
- Pause / Resume / Cancel。
- 播放进度追踪。
- 混合轮次控制。

验收标准：助手播放期间可以持续监听，正确处理主动打断、附和、噪声及句中停顿。

### Phase 3：Turn Adapter 研发

任务：

- 复现 Voice-Light Adapter。
- 建立数据处理流程。
- 自行训练轮次预测模型。
- 建立传统方法与学习模型基线。
- 完成消融实验及运行时集成。

验收标准：形成可复现训练代码、模型 Checkpoint、独立测试结果与错误分析。

模型未超过基线时，保留混合策略或原有基线作为产品默认方案。

### Phase 4：ASR / TTS 微调

任务：

- 完成基线模型 Bad Case 分析。
- 确定训练目标与数据来源。
- 实现模型微调。
- 完成微调前后评测。
- 部署验证后的模型权重。

验收标准：两个方向分别具备完整的数据处理、训练、评测和部署流程。

### Phase 5：性能优化

任务：

- 推测式生成。
- GPU 资源调度。
- 模型精度优化。
- 端到端延迟分析。
- 长时间运行测试。
- 发布桌面应用及算法评测报告。

验收标准：交付可运行系统、完整性能基准和可复现的实验结果。

---

## 11. 项目交付物

| 交付物 | 内容 |
|---|---|
| Desktop Application | Windows 桌面语音助手 |
| Voice Engine | 可独立接入 Agent 的语音交互模块 |
| ASR Training Pipeline | ASR 数据处理及微调代码 |
| TTS Training Pipeline | TTS 数据处理及微调代码 |
| Turn Adapter | 轮次预测模型及训练代码 |
| Benchmark Suite | 模型级与系统级评测工具 |
| Technical Documentation | 架构、接口、部署及开发说明 |
| Experiment Report | 对照实验、消融实验及错误分析 |

---

## 12. 项目边界

首版不涉及：

- 从零预训练大型 ASR 或 TTS 模型。
- 自研 LLM。
- 自研 Agent Runtime。
- 云端多租户语音服务。
- 移动端应用。
- 复杂长期记忆系统。
- 原生端到端 Speech-to-Speech 大模型训练。

主要开发资源集中于本地语音模型、全双工交互、模型训练及推理优化。

---

## 13. 项目风险与待验证事项

| 风险 | 应对方案 |
|---|---|
| 8GB 显存不足 | 优先使用在线 LLM；评估轻量 TTS、量化或 CPU 回退 |
| Nemotron 中间特征接口不兼容 | 优先复现 Voice-Light 的原始模型与接口 |
| 中文交互适配 | 首版英文验证，后续开展多语言模型适配 |
| TTS 流式生成兼容性 | 在 Phase 0 验证实际推理实现 |
| 人类对话训练数据受限 | 使用公开合成数据及合法授权的人类对话数据 |
| Adapter 未超过传统基线 | 保留混合策略，分析模型失效原因 |
| 扬声器回声导致误打断 | 优先验证 AEC，耳机模式作为初期回退 |
| Agent 工具不可取消 | 区分生成取消、执行取消及已完成操作 |
| 开源授权不兼容 | 固定版本并逐项核查许可证 |

---

## 14. 参考资料

**local-voice**

- Repository：https://github.com/mthomas100/local-voice
- Architecture：https://github.com/mthomas100/local-voice/blob/main/README.md
- Protocol：https://github.com/mthomas100/local-voice/blob/main/docs/PROTOCOL.md
- Spaces：https://github.com/mthomas100/local-voice/blob/main/docs/SPACES.md

**Voice-Light**

- Repository：https://github.com/BertilBraun/Voice-Light
- Paper：https://arxiv.org/abs/2609.20995
- Runtime：https://github.com/BertilBraun/Voice-Light/blob/master/docs/streaming-voice-agent-prototype.md
- Training：https://github.com/BertilBraun/Voice-Light/blob/master/docs/turn-taking-training.md
- Synthetic Dataset：https://huggingface.co/datasets/BertilBraun/voice-light-synthetic-audio

---

## 15. 最终产品定义

VoiceAgent 是一个以 Pi Agent 为任务执行核心、以本地流式 ASR 和 TTS 为语音基础、以可训练 Turn Adapter 为交互算法研究核心的全双工桌面语音助手。

其最终价值包含三个部分：

1. **产品价值：** 支持自然语音对话、实时打断和计算机任务执行。
2. **算法价值：** 支持 ASR、TTS、轮次预测模型的训练与优化。
3. **工程价值：** 实现消费级硬件上的实时推理、异步调度、资源管理与性能评测。

