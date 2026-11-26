# VoiceAgent 开发指南、工程规范与分阶段实施手册

本文档面向参与 VoiceAgent 桌面应用开发、本地语音推理优化及模型微调研究的开发者，说明本地开发环境搭建、Monorepo 协作命令、安全与隐私红线、可复现实验规范及 Phase 0–5 落地验收清单。

---

## 1. 开发环境准备（Windows 11 + NVIDIA GPU）

### 1.1 基础环境要求

| 软硬件依赖 | 最低要求 | 推荐开发配置 |
|---|---|---|
| **操作系统** | Windows 11 64-bit (22H2+) | Windows 11 23H2/24H2（可选配 WSL2 Ubuntu 24.04 用于大规模训练或 NeMo 完整工具链） |
| **GPU 与驱动** | NVIDIA GPU ≥ 8 GB VRAM | RTX 4060 / 4070 / 5070 (Laptop 或 Desktop)，最新 Game Ready / Studio 驱动（支持 CUDA 12.4 / 12.8+） |
| **Node.js 工具链** | Node.js `>= 22.0` + `pnpm >= 10.0` | Node.js `24.x` + `corepack enable` 激活 `pnpm` |
| **Python 工具链** | Python `3.11` / `3.12` + `uv` | `uv >= 0.6` 管理虚拟环境与锁文件 |
| **音频硬件** | 麦克风 + 扬声器/耳机 | Phase 0–1 联调建议备有耳机作为无回声基准对照；Phase 2 重点使用外放扬声器验证 AEC |

### 1.2 初始化步骤

1. **克隆仓库并准备环境变量**
   ```powershell
   git clone https://github.com/szaaaaaa/VoiceAgent.git
   cd VoiceAgent
   Copy-Item .env.example .env
   ```
   按需在 `.env` 中填入供 Pi Agent 调用的 LLM API Key（或确认本机已配置好 Pi CLI 全局凭证）。

2. **启用并安装 TypeScript 工作区依赖**
   ```powershell
   corepack enable
   pnpm install
   ```

3. **使用 `uv` 创建 Python 虚拟环境并安装基础与开发依赖**
   ```powershell
   uv venv --python 3.11
   uv pip install -e ".[inference,eval,dev]"
   ```
   > **提示**：安装 PyTorch 时请根据本机显卡架构指定官方 CUDA Wheel 源（例如 CUDA 12.6 / 12.8 源），安装完成后执行以下命令确认 GPU 可见性与 `bfloat16` 支持：
   > ```powershell
   > uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
   > ```

---

## 2. 日常开发与测试命令规范

| 操作场景 | 标准命令 | 说明 |
|---|---|---|
| **启动本地 Python 推理服务** | `uv run uvicorn inference.server:app --host 127.0.0.1 --port 8765` | 仅监听本机回环地址 |
| **启动 Electron 桌面端开发模式** | `pnpm dev` | 热重载启动 `apps/desktop` 并连接本地推理服务 |
| **全仓 TypeScript 类型检查** | `pnpm typecheck` | 对 `apps/*` 与 `packages/*` 执行严格模式检查 |
| **运行 TypeScript 单元测试** | `pnpm test` | 基于 `Vitest` 运行音频缓冲、状态机与权限测试 |
| **运行 Python 单元测试与代码检查** | `uv run ruff check . ; uv run pytest` | 执行代码风格静态检查与后端/评测测试 |

---

## 3. 工程质量、安全与算法研发红线

### 3.1 隐私与安全强制规范（对应 PRD §4.9, §8.3）
1. **严禁密钥与音频入库**：任何 `.env`、API Key、用户会话日志以及 `*.wav` / `*.pcm` 录音文件均不得提交至 Git。
2. **默认不落盘麦克风原始音频**：`VOICE_AGENT_SAVE_RAW_AUDIO` 默认为 `false`；仅当开发者主动开启数据采集模式并在 UI 显著位置提示录制状态时，才允许写入 `data/` 本地目录。
3. **本机监听隔离**：Python 推理服务默认且必须绑定 `127.0.0.1`，除非显式安全评审通过，否则禁止绑定 `0.0.0.0`。
4. **工具权限不可绕过**：所有具有写文件、删文件、执行命令或网络外发能力的工具必须注册进 `packages/agent/permissions.ts` 权限表；工作目录路径不等同于安全沙箱，必须校验规范化真实路径与参数白名单。

### 3.2 算法训练与评测可复现性规范（对应 PRD §6, §9.3）
1. **会话级数据隔离**：构建 `train` / `val` / `test` 时必须以完整对话 `session_id`（及说话人分组）为单位切分，严禁将同一段长对话切片后混入训练集和测试集造成声学或语境泄漏。
2. **严格因果性审计（Causality Check）**：Turn Adapter 及流式 ASR 特征提取严禁读取未来帧（Lookahead 超出流式 Chunk 边界外的帧）或双向全局池化；单元测试必须包含**“流式逐块输入 vs 整段截断输入输出完全一致”**的数值等价断言。
3. **测试集零调参**：所有概率阈值（如 `eot_probability_threshold`、`interruption_probability_threshold`）必须在 `val` 集确定并固化至配置文件，独立 `test` 集仅用于最终指标报告。
4. **完整保留负面结果**：若某次微调或 Adapter 结构改动在独立测试集上未超越传统基线（如 Smart Turn 或固定静音阈值），必须如实归档实验配置与误差分析，不得选择性删除负面实验或将其包装为正向收益。

---

## 4. 分阶段实施清单与验收门禁（Phase 0 – Phase 5）

### Phase 0：技术验证（Spike & Feasibility）
- [ ] 锁定参考项目 `local-voice` 与 `Voice-Light` 的参考 Commit 版本并完成开源许可证逐项核查（NeMo、Nemotron、Qwen3-TTS、Silero、Pi SDK）。
- [ ] 在目标 8 GB VRAM 设备上跑通 `nvidia/nemotron-speech-streaming-en-0.6b` 80ms 流式推理，验证多层 Encoder 中间特征实时提取 Hook。
- [ ] 跑通 `Qwen3-TTS` 0.6B 流式音频块生成，实测单句 TTFA 与 RTF。
- [ ] 压测 **Nemotron ASR + Causal GRU + Qwen3-TTS** 同进程常驻显存峰值，确认总占用稳定控制在 `≤ 8 GB` 且留有安全余量。
- **验收门禁**：输出 Phase 0 硬件实测基线报告，冻结基础模型与推理运行环境选型。

### Phase 1：基础语音 Agent（MVP Loop）
- [ ] 完成 `packages/protocol`、`packages/audio`（采集、16kHz 重采样、Ring/Pre-roll、基础播放队列）与 `inference/server.py` 联调。
- [ ] 完成 `packages/agent` 对 Pi SDK 的接入，支持多轮上下文、流式 `agent.delta`、可读文本净化及句级 `Sentence Buffer`。
- [ ] 实现 `Conversation` / `Action` 模式与 `Readonly` / `Ask` / `Trusted` 权限审批闭环。
- [ ] 完成 `apps/desktop` 主界面（支持 Hands-free、Push-to-talk、Text、Mute 四模式）。
- **验收门禁**：连续无故障完成 **≥ 20 轮**语音对话，支持文本混合输入与受权限拦截保护的本地工具调用。

### Phase 2：全双工语音交互（Full-Duplex & Barge-in）
- [ ] 集成播放参考回环的回声消除（AEC）与背景噪声抑制（NS），在外放扬声器条件下消除助手自激识别。
- [ ] 实现两阶段打断控制：阶段一 `≤ 250 ms`（P95）完成 Duck / Pause；阶段二 `≤ 500 ms`（P95）完成 Noise / Backchannel 恢复或 Interruption 取消。
- [ ] 实现 `playback.ack` 字符级播放游标追踪，打断后精准截断正式历史中的已听前缀并完整保留已执行工具上下文。
- [ ] 接入混合轮次控制器（VAD + 静音超时 + Smart Turn 基线 + 预留 Adapter 接口）。
- **验收门禁**：通过正常对话、长句中停顿、短暂附和、主动打断、咳嗽敲击噪声及外放回声 6 类核心交互测试。

### Phase 3：Turn Adapter 算法研发
- [ ] 搭建 `training/turn_adapter` 数据流水线（`Voice-Light Synthetic Audio`、`Mundo TurnBench Dev` 及授权对话数据）。
- [ ] 复现 `Voice-Light` 因果 GRU Adapter（`TURN-A3`），并完成自训练模型（`TURN-A4`）与混合交互控制（`TURN-A5`）对传统基线（`TURN-A0`~`A2`）的严格对照与消融实验。
- [ ] 将训练权重注册至 `models/registry.json` 并接入桌面开发者模式支持实时切换对比。
- **验收门禁**：交付可一键复现的训练脚本、模型权重、独立测试集对比表与 Bad Case 归因分析。

### Phase 4：ASR 与 TTS 领域/音色微调
- [ ] 完成基线 ASR 在目标口音、计算机操作指令与噪声场景下的 Bad Case 归因，执行 `ASR-A0` ~ `ASR-A3` 实验并验证通用集无显著退化。
- [ ] 以 `TTS-A1`（Zero-shot Voice Cloning）为强制基线，执行 `TTS-A2` / `TTS-A3` 说话人微调实验，评测 MOS、音色相似度、CER/WER 与 TTFA。
- **验收门禁**：ASR 与 TTS 各自形成从数据清洗、训练、自动评测到在线热加载的完整闭环。

### Phase 5：推理优化、推测式生成与长稳发布
- [ ] 实现 P2 推测式生成（`packages/interaction/speculative.ts`），量化评估端到端延迟收益与无效算力开销比。
- [ ] 完成推理精度（BF16/FP16/INT8）、Chunk Size 与 CUDA Stream 并发调优，冲击优化目标（E2E P50 `≤ 1 s`）。
- [ ] 执行 **≥ 4 小时**连续语音交互稳定性压测，排查延迟漂移、AudioBuffer 堆积与 GPU 显存碎片泄漏。
- **验收门禁**：交付可独立分发的 Windows 桌面端程序、完整评测基准套件与最终技术/实验报告。
