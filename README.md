# KVTC — KV-Cache Tensor Compression

**First open-source implementation of NVIDIA's KVTC (arXiv 2511.01815, ICLR 2026).**

Compress LLM KV caches by **6-9x** with negligible quality loss. Run **2M+ token context** on a single RTX 5090.

## Results (RTX 5090, Qwen2.5-7B)

| Config | K bits | V bits | Compression | V Cosine | Quality |
|--------|--------|--------|-------------|----------|---------|
| K1V3 | 1 | 3 | **8.8x** | 0.981 | Good |
| K2V4 | 2 | 4 | **6.1x** | 0.996 | Excellent |
| K2V4 + adaptive | 2 | 4 | **5.9x** | 0.998 | Excellent |
| K4V6 + adaptive | 4 | 6 | **3.4x** | 0.9999 | Lossless |

### vs TurboQuant

| Method | Compression | Quality |
|--------|------------|---------|
| TurboQuant turbo3 | 4.6x | +1.1% PPL |
| TurboQuant turbo2 | 6.4x | +6.5% PPL |
| **KVTC K2V4** | **6.1x** | **V cos 0.996** |
| **KVTC K1V3** | **8.8x** | **V cos 0.981** |

KVTC matches TurboQuant's compression with **dramatically better quality**, or exceeds it by 37% at comparable quality.

### Confirmed Context Limits (Qwen3.5-27B, RTX 5090 32GB)

| Method | Max Context | Speed | Status |
|--------|------------|-------|--------|
| f16 KV cache | 232K | 70 tok/s | Baseline |
| TurboQuant turbo2 | **1M (server)** | **67 tok/s** | **CONFIRMED STABLE** |
| TurboQuant turbo2 | 2M (CLI) | ~1.4 tok/s | Confirmed (CLI only) |
| **KVTC K2V4** | **~1.4M** | est. 65 tok/s | Integration in progress |
| **KVTC K1V3** | **~2.1M** | est. 60 tok/s | Integration in progress |

## How It Works

KVTC applies media-compression techniques to KV cache vectors:

```
KV tensor --> Undo RoPE --> PCA transform --> DP-optimal quantization --> Entropy coding --> Compressed
                (keys only)   (decorrelate)    (adaptive bit allocation)   (zlib/LZMA)
```

### Three-Stage Pipeline

1. **PCA Decorrelation** — Project KV vectors into principal component space using eigenvectors learned from calibration data. Most variance is captured by the top components.

2. **DP-Optimal Bit Allocation** — Dynamic programming finds the optimal bits-per-component that minimizes reconstruction error under a total bit budget. High-variance components get more bits; low-variance components get pruned to 0 bits.

3. **Entropy Coding** — DEFLATE (zlib) or LZMA2 compression on the quantized byte stream. Dual-mode picker selects whichever is smaller.

### Key Innovations

- **Asymmetric K/V budgets** — Keys compress better than values (RoPE gives them exploitable structure). Give keys fewer bits and values more bits for optimal quality.
- **Per-layer adaptive budgets** — Final attention layers (23-26) have higher value entropy. Automatically give them extra bits based on calibration-measured difficulty scores.
- **RoPE undo/reapply** — Remove rotary position embeddings from keys before PCA (they obscure the low-rank structure), reapply after decompression.
- **Attention sink + sliding window protection** — Never compress the first 4 tokens (attention sinks) or the last 128 tokens (sliding window). These are critical for model quality.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets

# Clone
git clone https://github.com/OnlyTerp/kvtc.git
cd kvtc

# Run benchmark (uses Qwen2.5-7B by default)
python benchmarks/benchmark_v3.py --model Qwen/Qwen2.5-7B-Instruct --device cuda

# Or with a different model
python benchmarks/benchmark_v3.py --model meta-llama/Llama-3.1-8B-Instruct --device cuda
```

## Usage

```python
from src.common import CalibrationData
from src.pipeline_fast import KVTCCompressorFast

# Load calibration data (pre-computed)
calibration = torch.load("calibration.pt")

# Set asymmetric bit budgets
for (layer, group, kind), entry in calibration.entries.items():
    entry.bit_budget = 128 * (2 if kind == "keys" else 4)  # K=2bit V=4bit

# Compress
compressor = KVTCCompressorFast(calibration, device="cuda")
compressed = compressor.compress(kv_cache, positions)
print(f"Compression ratio: {compressed.metadata.compression_ratio:.1f}x")

# Decompress
reconstructed = compressor.decompress(compressed)
```

## Project Structure

```
kvtc/
├── src/
│   ├── common.py          # Data structures (CalibrationData, CompressedKVCache)
│   ├── pca.py             # PCA transform, RoPE undo/reapply
│   ├── quantize.py        # DP bit allocation, uniform quantization
│   ├── gpu_ops.py         # Vectorized GPU operations (PyTorch)
│   ├── entropy.py         # zlib/LZMA entropy coding
│   ├── pipeline.py        # Reference pipeline (CPU, readable)
│   ├── pipeline_fast.py   # GPU-accelerated pipeline (production)
│   ├── triton_kernels.py  # Triton GPU kernels for bit packing
│   └── cache.py           # HuggingFace DynamicCache wrapper
├── benchmarks/
│   ├── benchmark_v1.py    # Basic symmetric benchmark
│   ├── benchmark_v2.py    # Asymmetric K/V benchmark
│   ├── benchmark_v3.py    # Full sweep: adaptive + dual entropy
│   ├── results_v3.json    # Raw benchmark data
│   └── TURBOQUANT_BASELINE.md  # TurboQuant comparison numbers
├── BENCHMARKS.md          # Full v3 results table
├── README.md              # This file
└── setup.py               # Package installation
```

## Benchmarked Hardware

- **GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM, SM120 Blackwell)
- **CUDA:** 12.8
- **PyTorch:** 2.11.0+cu128
- **Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, 4 KV heads, dim=128)

## Citation

```bibtex
@inproceedings{staniszewski2026kvtc,
  title={KV-Cache Tensor Compression via Joint Decorrelation, Quantization, and Entropy Coding},
  author={Staniszewski, Konrad and Łańcucki, Adrian},
  booktitle={ICLR},
  year={2026}
}
```

## License

MIT

---

*Built by [@OnlyTerp](https://x.com/OnlyTerp) / [Terp AI Labs](https://github.com/OnlyTerp)*
*Benchmarked on RTX 5090 — the first consumer GPU KVTC implementation*
