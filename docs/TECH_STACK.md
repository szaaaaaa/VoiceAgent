# VoiceAgent 技术栈选型与系统架构规范

本文档基于 [产品需求文档 (PRD V1.0)](live/VoiceAgent_PRD.md)，明确 VoiceAgent 全栈技术选型、版本基线、跨语言协作边界、消费级 GPU（8 GB VRAM）显存分配预算以及端到端低延迟设计准则。

---

## 1. 架构分层与语言边界决策

VoiceAgent 采用 **TypeScript（桌面交互与会话编排层） + Python（本地神经网络推理与算法研发层）** 的双运行时架构，两者在单机内通过仅绑定 `127.0.0.1` 的低延迟 **Local WebSocket** 通信。

### 1.1 为什么采用 TypeScript + Python 双层架构

| 维度 | TypeScript 层 (`apps/*`, `packages/*`) | Python 层 (`inference/*`, `training/*`, `evaluation/*`) |
|---|---|---|
| **核心职责** | Electron 桌面 GUI、实时音频采集与播放队列、全双工状态机编排、Pi Agent SDK 集成、工具权限审批 | 流式 VAD/ASR/TTS GPU 推理、Encoder 中间层特征提取、Causal Turn Adapter 推理、离线模型微调与基准评测 |
| **生态契合度** | 原生接入 **Pi Agent SDK**（TypeScript 编写）与 Web Audio API / `AudioWorklet`、React 高帧率可视化生态 | 原生支持 PyTorch / CUDA、NVIDIA NeMo（Nemotron Streaming）、Qwen3-TTS、Hugging Face 训练与评测工具链 |
| **故障隔离性** | 即使 Python 推理子进程发生 CUDA OOM 或异常重启，Electron 界面与 Pi Agent 会话状态依然存活，可无缝降级为纯文本交互 | 专注纯流式张量计算与模型生命周期管理，无需混杂复杂 GUI 状态、桌面快捷键或文件系统权限弹窗逻辑 |

### 1.2 为什么暂不强制全面迁移至 ONNX Runtime

PRD §7.1 明确规定：**V1 阶段以 PyTorch / CUDA 原生推理为主，后续按需评估 ONNX**。原因如下：

1. **Encoder 中间层特征无损提取**：Turn Adapter（参考 `Voice-Light`）需要从冻结的流式 Nemotron ASR Backbone 中实时读取多层中间表征（Hidden Representations）及跨块缓存状态（Cache Tensors），PyTorch Forward Hook 与原生模块接口具备最高灵活性。
2. **微调权重零转换闭环**：离线训练产出的 ASR（LoRA/全参）、Turn Adapter 与 Qwen3-TTS Checkpoint 可直接加载至在线推理后端验证，避免静态图导出阶段因动态流式 Cache 形状或自定义算子引发精度/行为偏差。
3. **局部轻量组件按需加速**：仅对接口完全固定、无中间特征依赖的独立算子（如 Silero VAD 或重采样滤波）允许选用 ONNX / TorchScript，核心语音主干保持 PyTorch 原生可控。

---

## 2. 全量技术栈矩阵与版本基线

### 2.1 桌面端与应用编排层（TypeScript / Node.js）

| 模块分类 | 技术选型 | 版本基线 | 选型依据与职责说明 |
|---|---|---|---|
| **包管理与工作区** | `pnpm` Workspace | `>= 10.x` | 严格依赖隔离，统一管理 `apps/desktop` 与 6 个内部功能包（`@voice-agent/*`） |
| **应用运行时** | Node.js + Electron | Node `>= 22 LTS` / Electron `>= 34` | Windows 11 原生桌面窗口、全局热键（Push-to-talk）、托盘驻留、Python 推理子进程守护 |
| **构建工具** | `electron-vite` + Vite | `>= 3.0` / Vite `>= 6.0` | 统一打包 Electron `main`、`preload` 与 `renderer`，支持毫秒级热更新 |
| **前端视图框架** | React + TypeScript | React `19.x` / TS `5.8+` | 严格类型安全；并发渲染保障转写流、工具状态卡片与 60fps 音频波形互不卡顿 |
| **前端状态管理** | `Zustand` | `>= 5.0` | 低开销精细订阅，独立维护采集轨（Listening）与输出轨（Thinking/Speaking）双状态及高频指标流 |
| **样式与组件** | Tailwind CSS + Radix UI | Tailwind `4.x` | 轻量无运行时 CSS 开销，快速构建主对话界面、高风险工具授权模态框与开发者监控 HUD |
| **音频采集与播放** | Web Audio API (`AudioWorklet`) | Chromium 内置 | 独立音频渲染线程处理帧采集、缓冲消费与样本级播放偏移统计，规避主线程 GC 抖动 |
| **音频信号前处理** | WebRTC Audio Processing (APM) / WASAPI | 原生绑定或 WASM + 系统回环 | 提供经过工业验证的声学回声消除（AEC3）、噪声抑制（NS）与自动增益控制，接收播放参考信号 |
| **智能体运行时** | **Pi Agent SDK** | `@mariozechner/pi-agent-core` / `@mariozechner/pi-ai`（兼容 `@earendil-works/*`） | 直接复用多提供商 LLM 抽象、上下文管理、多轮 Session 持久化与标准工具调用循环，不重复造轮子 |
| **协议校验** | `@sinclair/typebox` / `zod` | TypeBox `>= 0.34` | 与 Pi SDK 原生采用的 TypeBox 保持一致，对 WebSocket 事件与工具参数执行运行时零信任校验 |
| **单元测试** | `Vitest` | `>= 3.0` | 测试环形缓冲区、句级切分器、混合轮次状态机、打断裁决与权限拦截器 |

### 2.2 本地语音推理、训练与评测层（Python / CUDA）

| 模块分类 | 技术选型 | 版本基线 | 选型依据与职责说明 |
|---|---|---|---|
| **Python 环境与包管理** | Python + `uv` | Python `3.11`–`3.12` / `uv >= 0.6` | 极速解析锁定 CUDA 版 PyTorch 与科学计算依赖，隔离推理与训练虚拟环境 |
| **深度学习与加速** | PyTorch + CUDA | PyTorch `>= 2.6` / CUDA `12.4+`（RTX 50 系使用 CUDA `12.8+`） | 原生支持 `bfloat16` / `float16` 混合精度、SDPA 注意力加速与多 CUDA Stream 并发调度 |
| **本地通信服务** | `FastAPI` + `uvicorn` + `websockets` | FastAPI `>= 0.115` | 提供 `ws://127.0.0.1:8765` 全双工异步流式端点及 `/health`、`/models` 管理接口 |
| **数据契约校验** | `Pydantic` v2 | `>= 2.10` | 与 TypeScript 端协议严格镜像，导出标准 JSON Schema |
| **语音活动检测 (VAD)** | **Silero VAD** | `silero-vad >= 5.1` | 毫秒级 CPU/GPU 帧级语音概率预测（32ms 帧窗口），检测起止边界与播放期潜在插话 |
| **流式语音识别 (ASR)** | **NVIDIA Nemotron Streaming** | `nvidia/nemotron-speech-streaming-en-0.6b` (`nemo_toolkit[asr]`) | 0.6B Cache-Aware 流式架构，原生支持标点大小写、80ms 低延迟分块与多层 Encoder 特征提取 |
| **轮次预测与打断模型** | **Causal GRU Turn Adapter** | PyTorch 自定义模块（约 183K 参数） | 复现并扩展 `Voice-Light` 设计：因果单向 GRU 融合冻结 ASR 多层特征与播放状态，预测 `Completion` / `Floor Taking` / `Non-floor Feedback` |
| **流式语音合成 (TTS)** | **Qwen3-TTS** | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` / `CustomVoice` | 0.6B 参数量兼顾低显存与高自然度，基于 12Hz 多码本 Tokenizer 实现极低首包延迟（TTFA）流式生成及零样本克隆/微调 |
| **模型微调工具链** | Hugging Face `datasets` + `peft` + NeMo | PEFT `>= 0.14` | 支持 ASR LoRA/全参微调、Turn Adapter 序列标注训练、TTS 说话人微调与 TensorBoard / W&B 实验追踪 |
| **评测与科学计算** | `jiwer` + `scipy` + `scikit-learn` + `pytest` | `pytest >= 8.3` | 自动化计算 WER/CER、F1/False Cutoff Rate、ERLE、Speaker Similarity、RTF 与 P50/P95 延迟分布 |

> **Windows 11 + NVIDIA 平台兼容性提示**：
> - 若目标机器搭载新一代 Blackwell 架构显卡（如 RTX 5070 Laptop GPU, `sm_120`），需安装支持 CUDA 12.8 的 PyTorch 发行版。
> - NVIDIA `nemo_toolkit` 在纯 Windows 原生环境下部分编译扩展（如部分解码器依赖）若出现兼容受限，Phase 0 支持两种部署形态对比验证：① Windows 原生 Python（精简依赖加载模型权重）；② WSL2 Ubuntu CUDA 后端（通过本机回环 `127.0.0.1` 与宿主 Windows Electron 客户端无感通信）。两种形态共用完全一致的 Local WebSocket 协议，桌面端零代码改动。

---

## 3. 硬件资源与 8 GB 显存分配预算

PRD §8.1 将消费级 **8 GB VRAM NVIDIA GPU** 定为一等公民运行目标。为避免全双工并发（边听边说：ASR + Turn Adapter + TTS 同时在场）触发显存溢出（OOM），制定以下显存预算基线（以 `BF16` / `FP16` 推理精度测算）：

| 常驻组件 / 显存池 | 规模与精度 | 预估峰值显存占用 | 调度策略 |
|---|---|---:|---|
| **Windows DWM + Electron 渲染进程** | 系统桌面与 WebGL 波形 | ~0.6 – 0.9 GB | 限制后台硬件加速缓存，波形渲染采用轻量 2D/WebGL |
| **Nemotron Streaming ASR (0.6B)** | `BF16` 权重 + 流式 KV/Conv Cache | ~1.4 – 1.7 GB | **常驻显存**（`batch_size=1`），固定最大滚动上下文窗口防止缓存无限膨胀 |
| **Silero VAD + Causal Turn Adapter** | VAD + ~183K GRU 权重与状态 | < 0.1 GB | Silero VAD 可运行于 CPU 单线程（<1ms），Turn Adapter 常驻 GPU 紧贴 Encoder 输出 |
| **Qwen3-TTS (0.6B + 12Hz Tokenizer)** | `BF16` 权重 + 流式自回归 Cache | ~1.6 – 2.2 GB | **常驻显存**；句级合成完毕立即清理临时 KV Cache |
| **CUDA Context + PyTorch 内存分配器缓存** | cuBLAS / cuDNN Workspace + 碎片余量 | ~1.0 – 1.5 GB | 配置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，跨轮次按阈值执行轻量缓存回收 |
| **合计预估占用** | — | **~4.7 – 6.4 GB** | **保留 1.6 – 3.3 GB 安全运行余量（满足 ≤ 8 GB 硬指标）** |

### 3.1 为什么默认通过 Pi Agent 使用在线或外部进程 LLM
在 8 GB 显存条件下，若再强行同卡塞入 7B/8B 本地 LLM（即便 INT4 量化亦需 4.5–5.5 GB），必然与并发的流式 ASR + TTS 争夺显存与带宽，导致首包延迟恶化甚至 OOM。因此：
- **默认推荐配置**：本地 GPU 专职保障 **ASR + Turn Adapter + TTS** 的毫秒级实时流式体验；LLM 通过 Pi SDK 接入云端高性能模型或局域网独立算力节点。
- **降级保护**：当显存监控模块检测到可用显存低于安全水位（如外部程序抢占 GPU）时，支持：① 将 TTS 切换为 INT8 量化或轻量备选引擎；② 临时回退至纯文本交互模式并向用户发出明确状态提示。

---

## 4. 端到端实时性与延迟预算拆解

依据 PRD §8.1，端到端响应延迟严格定义为：**用户实际语音尾音结束时刻 $t_{\text{speech\_end}}$ 至扬声器播放出第一帧可听语音时刻 $t_{\text{audible\_play}}$**。

| 链路阶段 | V1 目标耗时 (P50) | 优化目标耗时 (P50) | 核心工程保障手段 |
|---|---:|---:|---|
| 1. 音频采集、重采样与本地 WS 传输 | ≤ 30 ms | ≤ 20 ms | `AudioWorklet` 20ms 帧直出，二进制裸 PCM 零 Base64 序列化开销 |
| 2. 流式 ASR 尾帧解码 + Turn Adapter 判定 | ≤ 280 ms | ≤ 160 ms | 80ms 流式分块增量编码（说话过程中已完成绝大部分编码），尾帧仅需增量收敛与因果分类 |
| 3. Pi Agent 首句可读文本生成 (LLM TTFT + 首句成型) | ≤ 900 ms | ≤ 450 ms | 流式 Delta 消费；P2 引入 Speculative Generation 在高置信度即将结束时提前预热生成 |
| 4. Sentence Buffer 切分 + Qwen3-TTS 首音频块合成 (TTFA) | ≤ 550 ms | ≤ 280 ms | 首句采用更激进的短子句切分阈值（如首标点 8–15 词即送入流式 TTS，后续句子恢复常规句长） |
| 5. 播放队列注入与声卡硬件缓冲输出 | ≤ 60 ms | ≤ 40 ms | 预热常开 `AudioContext` 输出节点，避免每次新建硬件流引发百毫秒冷启动爆音或延迟 |
| **端到端总延迟 ($t_{\text{audible\_play}} - t_{\text{speech\_end}}$)** | **≤ 1,820 ms (目标 ≤ 2 s)** | **≤ 950 ms (目标 ≤ 1 s)** | 全链路 `metrics` 逐段打点审计，区分排队耗时与纯计算耗时 |

### 打断响应延迟预算
- **阶段一（疑似人声快速避让 Duck / Pause）**：`Silero VAD` 连续命中阈值 + 基础能量门限触发，无需等待完整 ASR 文本，在 TypeScript 本地音频管线直接压低音量或暂停播放指针，确保 **P95 ≤ 250 ms**（优化目标 ≤ 150 ms）。
- **阶段二（打断确认与任务终止 Cancel / Abort）**：综合 `Turn Adapter` 的 `Floor Taking` 概率与增量 ASR 词汇特征，确认主动打断后立即清空待播队列并触发 `generation.cancel`，确保 **P95 ≤ 500 ms**（优化目标 ≤ 300 ms）。
