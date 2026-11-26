# Datasets & Audio Corpus Directory

此目录用于存放离线训练、微调与系统级基准测试所需的公开数据集、合成语音与经明确授权采集的对话数据。
根据隐私与安全规范（参见 `docs/live/VoiceAgent_PRD.md` §8.3），原始音频与数据文件不纳入 Git 版本控制。

## 推荐子目录约定

- `data/raw/`：原始公开数据集或经授权录音（只读归档）
- `data/processed/`：统一重采样（16 kHz 单声道 PCM）、切分与对齐后的特征/标注数据
- `data/splits/`：按完整会话（Session-level）严格划分的 `train` / `val` / `test` 清单（JSONL）
- `data/eval_sets/`：冻结的独立评测基准集（禁止用于训练或阈值调参）
