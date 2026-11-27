# VoiceAgent 开发、Pi 插件热挂载与微调实验指南（V2.0）

本文档面向日常开发与语音算法实验全流程，提供 Windows 11 + RTX 5070 本地环境规范、二次开发 Pi Agent 热插拔挂载步骤、**Google Drive / Hugging Face 零本地囤积数据集规范**、AutoDL 微调操作清单及阶段验收门禁。

---

## 1. 本地开发环境与常用命令速查

> **Windows PowerShell 5.1 终端注意**：旧版 PowerShell 不支持 `&&` 连接符与 `< $null` 输入重定向，串联命令请统一使用 `;` 并配合 `$LASTEXITCODE` 检查。

### 1.1 环境核验与服务启动

| 场景 | 标准命令（仓库根目录 `d:\voice-agent`） | 说明 |
|---|---|---|
| **验证 GPU 与 CUDA 12.8 环境** | `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"` | 确认输出 `2.11.0+cu128`、`RTX 5070`、`True` |
| **启动本地 Python 语音内核** | `.\.venv\Scripts\python.exe -m uvicorn inference.server:app --host 127.0.0.1 --port 8765` | 同时服务 Pi `/voice` 插件与桌面实验观测工作台 |
| **启动桌面算法实验观测工作台** | `npm run build -w @voice-agent/desktop; .\node_modules\electron\dist\electron.exe apps/desktop` | 或直接双击桌面 `VoiceAgent` 快捷方式打开观测台 |
| **TypeScript 全量类型检查与测试** | `npm run typecheck; npm test` | 校验共享包、`pi-voice` 插件与工作台代码 |
| **Python 静态检查与单元测试** | `.\.venv\Scripts\ruff.exe check .; .\.venv\Scripts\mypy.exe inference training evaluation; .\.venv\Scripts\pytest.exe` | 确保代码规范、类型注解与流式因果断言全部通过 |

---

## 2. 将 `/voice` 热插拔集成到本机二次开发 Pi Agent

无需修改本机二次开发 Pi Agent 的任何核心代码，支持以下两种挂载方式：

### 2.1 临时挂载调试（单次会话生效）
```powershell
pi -e d:/voice-agent/packages/pi-voice
```

### 2.2 持久路径软链安装（推荐日常使用）
```powershell
pi install d:/voice-agent/packages/pi-voice
```
- 该命令会将本地包路径写入 `~/.pi/agent/settings.json`（若仅希望在特定项目启用，加 `-l` 写入 `.pi/settings.json`）。
- 修改 `packages/pi-voice` 代码后，无需重新安装，直接在正在运行的 Pi 会话中输入 **`/reload`** 即可热重载最新逻辑。
- **常用交互指令**：
  - `/voice on` / `/voice off`：开启或关闭全双工免提语音模式；
  - `/voice ptt`：切换为按住/按键说话模式（快捷键 `Alt+V`）；
  - `/voice status`：查看当前连接的语音内核延迟与已激活的模型 Checkpoint；
  - `pi --voice`：启动 Pi 时直接激活语音监听。

---

## 3. 数据集零本地囤积规范（Google Drive + HF Streaming）

为保护本机 SSD 空间并保持代码仓库绝对轻量，严格执行以下数据铁律：

1. **禁止本机落盘原始数据集**：任何公开语料（LibriSpeech、AISHELL、Common Voice、Voice-Light Synthetic、TurnBench 等）均**不得下载解压至 `d:\voice-agent\data\`**。
2. **Hugging Face 在线流式读取（首选）**：训练与评测脚本统一通过 `training/storage.py` 开启 `streaming=True` 按需流式拉取音频并在内存完成特征提取：
   ```python
   from datasets import Audio, load_dataset

   ds = load_dataset(
       "BertilBraun/voice-light-synthetic-audio",
       split="train",
       streaming=True,
   ).cast_column("audio", Audio(sampling_rate=16000))
   ```
3. **Google Drive 云盘承载自采录音与大模型权重**：
   - 将个人授权音色微调语料、整理好的领域指令测试集及归档 Checkpoint 存放于 Google Drive（本机已安装 Google Drive 桌面端或通过 `rclone` 同步）；
   - 在 `.env` 中设置 `VOICE_AGENT_DATA_ROOT="G:/My Drive/VoiceAgentData"`，所有训练与评测脚本自动从该路径读写外部资产；
   - Git 仓库仅跟踪代码、`configs/*.yaml`、`models/registry.json` 元数据以及 `docs/experiments/` 下的实验指标报告。

---

## 4. AutoDL 云端微调省钱与防泄漏操作规范（¥100 预算闭环）

针对耗时的全量/多轮 ASR LoRA 与 Qwen3-TTS 说话人微调任务，在 AutoDL 单卡 **RTX 4090 (24GB)** 按量计费实例（约 ¥2.2–2.6/h）执行，总预算严格控制在 **¥30–¥50** 内：

1. **上云前本地 100% 冒烟通过**：所有训练脚本必须先在本机 RTX 5070 上用 `max_steps=5` 流式小样本跑通前向、反向、评估与 Checkpoint 导出，严禁在计费实例上现场改低级 Bug。
2. **数据与权重直通云盘**：利用 AutoDL 学术加速 + Hugging Face `streaming=True` 流式拉取公开语料，通过 `rclone` 或 Google Drive 仅拉取轻量自建微调集并回传最终 LoRA / Adapter 权重（通常仅数十 MB）。
3. **强制自动关机守护**：每次启动长训练任务必须追加完成即关机保护，杜绝空跑扣费：
   ```bash
   python -m training.asr.run_lora_autodl --config configs/asr_lora.yaml ; shutdown -h now
   ```
4. **科学实验四大门禁**：
   - **会话隔离**：严格按 `session_id` / `speaker_id` 切分 `train / val / test`；
   - **因果校验**：任何涉及 `CausalTurnGRU` 或流式 ASR 特征提取的改动必须通过 `verify_streaming_causality` 单测（流式逐块 vs 整段输入误差 `≤ 1e-5`）；
   - **强基线对照**：TTS 微调必须与 `3s Zero-shot Voice Cloning` 同场盲测对比；ASR 微调必须同时报告通用测试集 WER 以证明无灾难性遗忘；
   - **如实归档**：无论正向提升还是负面结论，均通过 `evaluation/` 生成不可变 JSON Manifest 归档至 `docs/experiments/`。

---

## 5. V2 后续开发与实验推进路线（Implementation & Experiment Roadmap）

### Milestone 1：Pi `/voice` 热插拔扩展与 Python 原生音频守护落地（P0）
- [ ] 实现 `inference/audio_io.py`（WASAPI 16kHz 采集、24kHz 播放队列、Duck/Pause/Resume/Cancel 与精确硬件播放游标上报）及 `server.py` 的 `Controller + Observer` 广播协议。
- [ ] 新建 `packages/pi-voice` 标准 Pi Package，对接 `ExtensionAPI`（`/voice`、`Alt+V`、`message_update` 流式净化播报、`ctx.abort()` 语音打断、`before_agent_start` 已听前缀对齐）。
- [ ] 在本机二次开发 Pi Agent 中完成 `pi install d:/voice-agent/packages/pi-voice` 真机联调，实现连续 ≥20 轮免提对话与实时打断。

### Milestone 2：桌面算法实验与观测工作台升级（P0–P1）
- [ ] 将 `apps/desktop` 升级为四大专业实验视图：Observer 实时信号示波台、Checkpoint 热插拔面板、ASR/TTS 多模型 A/B 盲测实验台、实验报告与 Bad Case 回放器。
- [ ] 验证终端运行 Pi `/voice` 对话时，桌面工作台零干扰实时同步绘制 VAD 与 `CausalTurnGRU` 三头概率动态曲线。

### Milestone 3：流式 TTS 加速与 RTX 5070 全双工性能优化（P1）
- [ ] 搭建 WSL2 CUDA `FlashAttention-2` + `torch.compile` 推理环境并接入极速备选 TTS 引擎，将 `Qwen3-TTS-0.6B` 实时率从 Windows 原生 `RTF=5.01` 压降至 `< 0.85`（首包 `TTFA ≤ 350 ms`）。
- [ ] 输出优化前后 6 项核心指标实测对比报告（`OPT-01 ~ OPT-03`）。

### Milestone 4：ASR LoRA 微调、Turn Adapter 真人对话适配与 TTS 音色微调交付（P1）
- [ ] 基于云盘/流式语料完成中英混合技术口令 ASR LoRA 微调（`ASR-A0 ~ ASR-A3`），修复诸如 `"RTX 5070" → "RTX 577"` 等技术术语 Bad Cases 并通过通用集防遗忘门禁。
- [ ] 完成 `Qwen3-TTS` 目标说话人微调与 3s Zero-shot 强基线对照评测（`TTS-A0 ~ TTS-A3`），在桌面工作台完成双盲 MOS 评估。
- [ ] 扩展 `CausalTurnGRU` 在真实双通道对话/实采全双工打断样本上的微调评测（`TURN-A0 ~ TURN-A5`），固化全套可复现权重注册表与最终实验汇总报告。
