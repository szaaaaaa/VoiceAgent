# Local Models & Checkpoints Directory

此目录用于存放本地推理所需的预训练模型缓存与离线微调产出的 Checkpoint 注册表。
为避免大体积权重文件进入 Git 仓库，除说明文件外的所有权重文件（`*.pt`、`*.safetensors`、`*.nemo`、`*.onnx` 等）均已被 `.gitignore` 忽略。

## 推荐子目录约定

- `models/hf_cache/`：Hugging Face 模型缓存（`HF_HOME`）
- `models/asr/`：Nemotron Streaming 基础权重与微调 Checkpoint
- `models/tts/`：Qwen3-TTS 0.6B 基础权重、音色参考音频特征与微调权重
- `models/turn_adapter/`：Causal GRU Turn Adapter 训练权重与阈值校准元数据
- `models/registry.json`：本地模型版本注册清单（记录 Checkpoint 哈希、训练实验 ID 与精度配置）
