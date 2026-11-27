# RTX 5070 Laptop 8GB 四模型同卡真实闭环实测报告

- **GPU：** `NVIDIA GeForce RTX 5070 Laptop GPU`（8,151 MiB 总显存）
- **同卡加载模型：** `silero-vad-onnx` + `transformers:openai/whisper-base.en` + `qwen3-tts-cuda-sdpa` + `CausalTurnGRU (causal_gru_best)`

## 真实闭环实测指标

| 指标 | 实测值 | 说明 |
|---|---:|---|
| **四模型同卡峰值显存 (Peak VRAM)** | **2402.4 MB** | 8GB 显存剩余余量 **5748.6 MB** |
| **TTS 热启动 RTF (BF16 + SDPA)** | **5.006** | 生成 6.8s 24kHz 语音耗时 34.04s（对比冷启动 Eager RTF=5.746） |
| **真实 Silero VAD 最高语音概率** | **1.0** | 对 Qwen3-TTS 真实生成语音的神经 VAD 命中率 |
| **真实 GPU ASR 实时率 (RTF)** | **0.0417** | 处理 6.8s 语音耗时 0.283s |
| **TTS → ASR 闭环回环 CER / WER** | **0.0556 / 0.3333** | 原文：`Hello, VoiceAgent is running locally on your RTX 5070.` → 识别：`Hello, Voice Agent is running locally on your RTX 577.` |
| **Turn Adapter 单步 GPU 延迟** | **32.806 ms** | CausalTurnGRU 实时帧推理耗时 |
