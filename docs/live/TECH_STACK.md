# VoiceAgent 技术栈选型、运行环境与性能预算规范（V2.0）

本文档基于 [产品需求文档 (PRD V2.0)](VoiceAgent_PRD.md)，明确重构后「**Python 语音训练推理内核 + 热插拔 Pi `/voice` 插件 + 桌面实验观测工作台**」三支柱架构的技术选型、版本基线、跨环境依赖隔离策略、RTX 5070 8GB 实测显存预算及端到端低延迟优化方案。

---

## 1. 关键架构决策（Architectural Decisions）

### 1.1 为什么将智能体入口重构为 Pi Package 扩展而非独立 Agent 应用
1. **零重复建设，原汁原味复用二次开发成果**：本机已部署二次开发的 Pi Agent（基于 `@earendil-works/pi-coding-agent` v1.4.0），具备完整的模型提供商配置、上下文压缩、自定义工具与工作流。通过标准 `pi-package` 扩展机制挂载 `/voice`，无需在外部维护第二套会话状态或工具权限系统。
2. **jiti 原生 TypeScript 热加载**：Pi 宿主通过 `jiti` 直接加载扩展源码（`packages/pi-voice/src/index.ts`），支持 `pi install d:/voice-agent/packages/pi-voice` 本地路径软链挂载与会话内 `/reload` 秒级热更新，开发迭代零打包等待。

### 1.2 为什么在 Python 内核引入原生 WASAPI 音频守护（双模式音频架构）
在 V1 架构中，麦克风采集与扬声器播放依赖 Electron Chromium 内部的 `AudioWorklet`。当语音模式变成运行于终端（CLI/TUI）的 Pi `/voice` 插件时，宿主处于纯 Node.js 环境，没有浏览器 Web Audio API：
- 若在 Node.js 侧引入原生 C++ 音频模块（如 `naudiodon`），极易在 Windows + Node 24 环境下引发 `node-gyp` 编译失败与 ABI 版本冲突。
- **最优工程解（Ponytail 最小完备方案）**：由本机常驻的 **Python 语音内核（`inference/audio_io.py`，基于 `sounddevice` / PortAudio WASAPI）** 直接接管低延迟麦克风采集（16kHz）与扬声器队列播放（24kHz）：
  1. `packages/pi-voice` 保持 **100% 纯 TypeScript 零原生二进制依赖**，仅通过本地 WebSocket 交互轻量 JSON 事件（转写、流式短句、打断指令、播放游标）；
  2. 采集帧与播放参考帧在 Python 进程内零拷贝直通 AEC、Silero VAD 与 GPU 模型，省去跨进程序列化与 IPC 抖动，同时由声卡硬件回调精确统计已播放采样点数（映射为字符级 `PlaybackCursor`）；
  3. 同时保留 WebSocket 二进制裸 PCM 通道（`ws_pcm` 模式），供 **桌面实验观测工作台（`apps/desktop`）** 进行音频文件注入回放、A/B 试听与浏览器端调试。

### 1.3 Windows CUDA 12.8 与 WSL2 / AutoDL 训练环境解耦策略
实机验证表明，NVIDIA Blackwell 架构（RTX 5070 Laptop，`sm_120`）需要 `torch>=2.11.0+cu128`，而 `nemo_toolkit[asr]` 在纯 Windows 环境下与新版 `torch+cu128` 及 `qwen-tts` 依赖的 `transformers==4.57.x` 存在包版本锁冲突，且 `flash-attn` 官方缺少预编译 Windows Wheel。因此采用清晰的**双环境分工**：

| 运行环境 | 适用任务 | 核心模型组合 | 通信方式 |
|---|---|---|---|
| **Windows 11 原生 `.venv` (`cu128`)** | 日常本地一键启动、工作台观测、多语种 Whisper + TurnGRU + TTS 闭环 | `silero-vad-onnx` + `transformers` (`whisper-*` + PEFT LoRA) + `CausalTurnGRU` + `qwen-tts` | `127.0.0.1:8765` 本机回环 |
| **WSL2 Ubuntu 24.04 / AutoDL 容器** | `Nemotron-0.6B` 原生流式研究、`FlashAttention-2` + `torch.compile` 极速 TTS、LoRA 与音色微调训练 | `nemo_toolkit[asr]` (`nemotron-speech-streaming-en-0.6b`) + `qwen-tts` (`flash-attn`) + `peft` | WSL2 与宿主机天然共享 `127.0.0.1:8765`，Pi 插件与桌面工作台**零代码改动无缝切换** |

---

## 2. 全栈技术矩阵与实测版本基线

### 2.1 Python 语音训练、推理与评测内核（Layer 1）

| 组件 | 选型与实测锁定版本 | 核心职责 |
|---|---|---|
| **Python & 深度学习框架** | Python `3.11.15` + `torch==2.11.0+cu128` / `torchaudio==2.11.0+cu128` | 原生支持 RTX 5070 Blackwell `sm_120`、`bfloat16` 与 SDPA 注意力 |
| **原生音频 I/O 与回声处理** | `sounddevice` (PortAudio WASAPI) + 播放参考谱减/自适应滤波 AEC | 20ms 低延迟采集、24kHz 无缝播放队列、Duck/Pause/Resume/Cancel 与样本级游标追踪 |
| **本地通信网关** | `FastAPI` + `uvicorn` + `websockets` + `Pydantic v2` | 提供 `ws://127.0.0.1:8765/ws/v1/voice` 控制/数据双通道及实验管理 REST 接口 |
| **语音活动检测 (VAD)** | `silero-vad==6.3` (`onnxruntime`) | 32ms 窗口帧级人声概率检测（CPU 单步耗时 `< 1 ms`） |
| **流式语音识别 (ASR)** | `nvidia/nemotron-speech-streaming-en-0.6b`（WSL2/AutoDL）<br>`openai/whisper-base` / `whisper-small` + LoRA（Win 原生多语种） | 流式 Partial/Final 识别 + 冻结 Encoder 多层因果特征提取 |
| **全双工轮次与打断模型** | 自研 `CausalTurnGRU`（PyTorch，约 34K–315K 参数，默认 2×128 = `215,686` 参数） | 融合 Encoder 特征与播放状态，实时输出 `Completion` / `Floor Taking` / `Backchannel` 三头概率 |
| **流式语音合成 (TTS)** | `Qwen/Qwen3-TTS-12Hz-0.6B` (`qwen-tts==0.1.1`) + 快速低延迟备选引擎 | 本地音色克隆、少样本说话人微调加载与句级流式合成 |
| **微调与云端流式数据** | `peft==0.22.0` + `transformers==4.57.3` + `datasets` (`streaming=True`) + Google Drive | 零本地数据集囤积；支持 ASR LoRA 微调与 TTS 说话人微调 |
| **科学评测与质量保障** | `jiwer==4.0.0` + `pytest` + `ruff` + `mypy` | 自动化计算 WER/CER、EOT F1/误截断率、因果等价断言与 GPU 性能快照 |

### 2.2 热插拔 Pi `/voice` 插件与共享核心包（Layer 2）

| 组件 | 选型与版本基线 | 关键约束说明 |
|---|---|---|
| **宿主扩展 SDK** | `@earendil-works/pi-coding-agent` (`^1.4.0`)、`@earendil-works/pi-ai`、`@earendil-works/pi-tui`、`@sinclair/typebox` | **铁律**：全部声明在 `peerDependencies: { "*": "*" }`，严禁放入 `dependencies` 以免破坏宿主单例上下文 |
| **共享协议与状态机** | `@voice-agent/protocol`、`@voice-agent/speech`、`@voice-agent/interaction`、`@voice-agent/metrics` | 纯 TypeScript 实现（Node `24.x` / TS `5.8`），提供三级 ID 过滤、可读文本净化器、自适应分句器与三账本中断对齐 |
| **单元测试** | `Vitest 3.2+` | 覆盖分句净化、6 类全双工轮次状态机、播放游标截断与扩展钩子模拟测试 |

### 2.3 桌面算法实验与观测工作台（Layer 3）

| 组件 | 选型与版本基线 | 核心职责 |
|---|---|---|
| **桌面容器与构建** | `Electron 45.5` + `Vite 7.3` + `React 19` + `TypeScript 5.8` | 独立实验工作台窗口；通过 `session.clearCache()` 与本地 FastAPI 静态挂载杜绝旧前端缓存残留 |
| **可视化与评测交互** | Canvas 60fps 示波器 + Checkpoint 热插拔控制台 + A/B 盲测播放器 | 实时绘制波形/VAD/GRU 三头概率曲线、对比多版本微调模型输出并浏览 `docs/experiments/` 报告 |

---

## 3. RTX 5070 8GB 显存实测基线与分配预算

基于本机 **NVIDIA GeForce RTX 5070 Laptop GPU（总显存 8,151 MiB）** 的四模型同卡闭环实测数据（详见 [rtx5070_real_models_benchmark.md](../experiments/rtx5070_real_models_benchmark.md)），显存分配与安全水位如下：

| 常驻模块 | 精度与运行设备 | 实测 / 预算显存占用 | 调度与防溢出策略 |
|---|---|---:|---|
| **Silero VAD v6** | FP32 ONNX (CPU) | `0 MB` GPU | CPU 推理足够快（`<1ms`），零显存占用并避免小算子抢占 GPU 队列 |
| **流式 ASR (`Whisper-base` / `Nemotron-0.6B`)** | `BF16` (CUDA) | `~350 – 1,550 MB` | 常驻显存，`batch_size=1`，限制滚动流式 Cache 最大窗口长度 |
| **`CausalTurnGRU` (`causal_gru_best.pt`)** | `BF16` / `FP32` (CUDA) | `< 15 MB` | 21.5 万参数极轻量常驻，紧贴 Encoder 特征同设备推理 |
| **`Qwen3-TTS-12Hz-0.6B` + Tokenizer** | `BF16` SDPA / FA2 (CUDA) | `~1,850 – 2,100 MB` | 常驻显存；每次句级合成结束或收到 `generation.cancel` 立即清理临时自回归 Cache |
| **四模型同卡实测峰值（Whisper + Qwen3-TTS + GRU + VAD）** | `BF16` | **2,402.4 MB** | **实测剩余可用显存 5,748.6 MB（占总显存仅 29.5%）** |
| **切换 `Nemotron-0.6B` + 工作台 GUI 预留后总峰值上限** | `BF16` | **≤ 4,800 MB** | 仍保留 **> 3.2 GB** 安全余量，支持长时全双工并发零 OOM |

> **为什么坚持 LLM 不上本机 8GB 显卡**：Pi Agent 所调用的 Coding LLM 需要长上下文（32K–200K Window）与强工具推理能力，继续由宿主 Pi 路由至云端或独立算力节点，使本机 8GB RTX 5070 100% 服务于低延迟流式语音模型与本地微调验证。

---

## 4. 端到端实时性预算与 TTS 瓶颈专项攻关

端到端响应延迟严格定义为：**用户语音尾音结束时刻 $t_{\text{speech\_end}}$ 至扬声器硬件播出首帧可听语音时刻 $t_{\text{audible\_play}}$**。

### 4.1 全链路延迟预算拆解

| 阶段 | 当前状态 / 瓶颈诊断 | V2 验收目标 (P50) | 优化目标 (P50) | 关键工程与算法手段 |
|---|---|---:|---:|---|
| **1. 声卡采集与帧送达** | WASAPI 20ms 帧直连 | ≤ 25 ms | ≤ 15 ms | Python 本地 `sounddevice` 直采省去浏览器重采样与跨进程音频搬运 |
| **2. 流式 ASR 收敛 + CausalGRU EOT 裁决** | ASR `RTF=0.0417`，GRU 单步 `0.02–32ms` | ≤ 240 ms | ≤ 150 ms | 说话期增量编码，尾音后由 `CausalTurnGRU` 替代传统 700ms 死等静音阈值 |
| **3. 宿主 Pi Agent 首短句生成** | 取决于宿主所选 LLM TTFT | ≤ 850 ms | ≤ 450 ms | `message_update` 流式监听 + 首标点激进短句切分（8–15 字即送合成） |
| **4. 流式 TTS 首音频块生成 (TTFA)** | **当前核心瓶颈**：Win 原生无 FA2 下 `Qwen3-TTS` `RTF=5.01` | **≤ 550 ms** | **≤ 260 ms** | 见 §4.2 专项加速路线 |
| **5. 声卡缓冲注入与首帧播放** | 常开输出流预热 | ≤ 45 ms | ≤ 30 ms | 常驻预热 `OutputStream`，消除声卡冷启动咔哒声与初始化延迟 |
| **端到端总计 ($t_{\text{audible\_play}} - t_{\text{speech\_end}}$)** | — | **≤ 1,710 ms** | **≤ 905 ms** | 全程由 `metrics` 区分计算耗时与队列等待耗时 |

### 4.2 流式 TTS 延迟与实时率专项优化路线（针对实测 `RTF = 5.01`）
1. **WSL2 Linux + `FlashAttention-2` + `torch.compile`（主攻方向）**：在 WSL2 Ubuntu CUDA 12.8 环境中启用 `Qwen3-TTS` 官方推荐的 `flash_attention_2` 与静态 KV Cache 图编译（参考 `faster-qwen3-tts` CUDA Graph 思路），消除 Python 逐 Token 自回归调度开销，目标将 `0.6B` 模型 `RTF` 从 `5.01` 压降至 **`0.35 – 0.75`**，并利用流式 Chunk 解码使首包 `TTFA ≤ 300 ms`。
2. **双引擎热切换策略（研发模式 vs 极速生产力模式）**：
   - **音色微调与高自然度模式**：挂载 `Qwen3-TTS-12Hz-0.6B`（支持 Zero-shot 克隆与 `TTS-A2` 微调 Checkpoint）；
   - **极速日常编码响应模式**：保留毫秒级轻量流式 TTS 后端（如 `kokoro-onnx` / 本地系统神经语音引擎，`TTFA < 80 ms`，`RTF < 0.08`），允许在 Pi `/voice` 中通过 `/voice engine qwen3|fast` 一键切换，并在实验报告中客观呈现两者的音色自然度 vs 延迟权衡曲线。
