# TurboQuant on RTX 5090 (Blackwell SM120) — First CUDA Benchmarks

**First known public TurboQuant CUDA benchmark on NVIDIA Blackwell (SM120) hardware.**

## UPDATE (2026-03-31): Server Mode Confirmed

TurboQuant turbo2 now running as a **persistent server** via llama-server:

| Config | Context | Speed | VRAM | Status |
|--------|---------|-------|------|--------|
| turbo2, `-c 1000000` | **1,000,000 tokens** | **67.2 tok/s** | 17.3 GB | **STABLE** |
| turbo2, `-c 262144` | 262K tokens | 67.7 tok/s | 15.9 GB | STABLE |
| turbo2, `-c 32768` | 32K tokens | 68.6 tok/s | ~16 GB | STABLE |
| turbo2, `-c 2000000` (CLI) | **2,000,000 tokens** | ~1.4 tok/s | ~30 GB | Works (CLI only) |

**QwOpus (Qwen3.5-27B-Q4_K_M, Opus-distilled) serving 1M context at 67 tok/s on a single RTX 5090.**

Server command:
```
llama-server.exe -m Qwen3.5-27B-Q4_K_M.gguf -ctk turbo2 -ctv turbo2 -fa on -ngl 99 -c 1000000 --host 0.0.0.0 --port 8081 -np 1
```

---

## Hardware
- **GPU:** NVIDIA GeForce RTX 5090, 32GB VRAM, Compute Capability 12.0
- **Driver:** 591.86, CUDA 12.8 (build), CUDA 13.1 (driver)
- **Build:** seanrasch/llama-cpp-turboquant fork, MSVC 2022, Ninja, SM120a

## Model
- **Qwen3.5-27B Q4_K_M** (15.58 GiB, 26.90B params) by Unsloth
- 36 layers, 4 KV heads, head_dim 128 (256 for some layers), GQA

## Build Notes (Windows + CUDA + SM120)
The CUDA kernels (turbo-quant.cuh, turbo-wht.cu, turbo-innerq.cu) compiled and ran on SM120 **without modification**. Three minor Windows DLL linking patches were needed:

1. `#define _USE_MATH_DEFINES` before `<math.h>` in `ggml-turbo-quant.c` — MSVC doesn't define `M_PI` by default
2. Made `turbo3_cpu_wht_group_size` a local definition in `ops.cpp` instead of cross-DLL extern
3. Replaced `#ifdef GGML_USE_CUDA` extern block in `llama-kv-cache.cpp` with local stubs — Windows DLLs can't resolve cross-DLL extern symbols at link time without explicit `__declspec(dllimport)`

## Prefill Benchmarks (tok/s) — turbo3

| Context | f16 KV | turbo3 KV | turbo3 vs f16 |
|--------:|-------:|----------:|--------------:|
| 512 | 3,534 | **3,541** | 1.00x |
| 2,048 | 3,516 | **3,575** | **1.02x** |
| 8,192 | 3,291 | **3,470** | **1.05x** |
| 32,768 | 2,482 | **3,068** | **1.24x** |
| 65,536 | OOM | **2,498** | ∞ |
| 131,072 | OOM | **1,731** | ∞ |

**turbo3 is faster than f16 at every context length** for prefill, because the compressed cache uses less memory bandwidth. The advantage grows with context: +24% at 32K.

## Generation Benchmarks (tok/s)

| KV Type | tg128 | vs f16 |
|---------|------:|-------:|
| f16 | 70.22 | baseline |
| turbo3 | 67.77 | 0.965x |
| turbo2 | **71.1** | **1.01x** |

turbo2 generation is actually **faster than f16** — the 2-bit cache has even less bandwidth overhead.

## The Flex: 1.5 Million Token Context

Confirmed working: **1,500,000 tokens of context** on a single RTX 5090 with Qwen3.5-27B using turbo2 (2-bit, 6.4x compression).

```
llama-cli -m Qwen3.5-27B-Q4_K_M.gguf -ctk turbo2 -ctv turbo2 -fa on -ngl 99 -c 1500000
```

The model loaded, generated coherent output, and ran at:
- Prompt: 7.8 tok/s (initial tokens — expected at 1.5M context)
- Generation: 1.4 tok/s (expected — attention over 1.5M positions)

**For comparison:** f16 KV cache OOMs at ~232K context on the same GPU. turbo2 extends this to **1.5M** — a **6.5x increase**.

## 1M Context Performance (turbo2)

At 1,000,000 token context:
- Prompt: 122.8-130.1 tok/s
- Generation: 69.3 tok/s
- All 36 layers on GPU, model fully offloaded

## Theoretical Context Limits (Qwen3.5-27B on RTX 5090)

| KV Cache | Compression | Max Context | Notes |
|----------|------------|------------|-------|
| f16 | 1.0x | 232K | Baseline |
| q8_0 | 1.9x | 436K | Near-identical quality |
| q4_0 | 3.6x | 823K | Math accuracy degrades |
| turbo3 | 4.6x | **1.1M** | +1.1% PPL |
| turbo2 | 6.4x | **1.5M** | +6.5% PPL |

## vs Previous Approach (Ollama Q4_0 KV)

Previously running QwOpus at 1M via Ollama with `OLLAMA_KV_CACHE_TYPE=q4_0`:
- 63 tok/s generation, 29.8 GB VRAM (barely fits)
- Q4_0 has known math quality issues (simple scalar quantization)

Now with TurboQuant turbo2 via llama.cpp:
- **69.3 tok/s** generation (+10%), **~18 GB** VRAM (14+ GB free)
- WHT rotation + Lloyd-Max optimal quantization (mathematically principled)
- Better quality at even better compression

## Key Takeaway

TurboQuant CUDA kernels work on Blackwell (SM120) out of the box. The compressed KV cache is **faster** than f16 for prefill at every context length because it uses less memory bandwidth. Generation overhead is negligible (3.5% for turbo3, actually faster for turbo2).

The RTX 5090 can run a 27B model with **1.5M context** using turbo2 — something that would require ~96 GB of VRAM with f16 KV cache.

---

*Benchmarked 2026-03-31 by [@OnlyTerp](https://x.com/OnlyTerp) / Terp AI Labs*
*Build: seanrasch/llama-cpp-turboquant @ 7b750787b (8664)*
*Windows patches available on request — happy to PR*
