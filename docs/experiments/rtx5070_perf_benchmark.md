# RTX 5070 Laptop 8GB 本地推理性能实测对比

- **是否加载完整重型权重 (`load_heavy_models`)：** `False`
- **CUDA 可用：** `True` (NVIDIA GeForce RTX 5070 Laptop GPU)

| 指标 | 优化前 (`before_fp32_eager`) | 优化后 (`after_bf16_cached_stream`) | 变化 Delta |
|---|---:|---:|---:|
| ASR RTF | 0.03901 | 0.04897 | 0.00996 |
| TTS TTFA (ms) | 1.26 | 1.32 | 0.06 |
| E2E Latency P50 (ms) | 181.27 | 181.25 | -0.02 |
| E2E Latency P95 (ms) | 181.46 | 181.57 | 0.11 |
| Barge-in Latency P95 (ms) | 18.0 | 18.0 | 0.0 |
| Peak VRAM (MB) | 26.7 | 26.3 | -0.4 |
| False Interruption Rate | 0.08 | 0.03 | -0.05 |
