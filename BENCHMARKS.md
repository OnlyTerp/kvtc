# KVTC v4 Benchmark -- All High-Impact Optimizations
## Qwen/Qwen2.5-7B-Instruct on NVIDIA GeForce RTX 5090

### Optimizations Applied
1. **Fused PCA+Quantize** -- single GPU pass, no intermediate tensor allocation
2. **Entropy-based adaptive budgets** -- per-layer bit allocation based on eigenvalue spectrum entropy
3. **ANS entropy coding** -- rANS (range Asymmetric Numeral Systems) vs zlib/LZMA triple-pick
4. **Per-layer K/V split** -- independent key and value budgets per layer

### Results

| Config | K | V | Opts | Ratio | K Cos | V Cos | Compress | Decompress | Quality |
|--------|---|---|------|-------|-------|-------|----------|------------|---------|
| K2V4-baseline | 2 | 4 | baseline | **5.9x** | 0.9968 | 0.9972 | 287ms | 5536ms | Excellent |
| K1V3-baseline | 1 | 3 | baseline | **8.9x** | 0.9924 | 0.9866 | 255ms | 4791ms | Good |
| K2V4-adaptive | 2 | 4 | adaptive | **5.9x** | 0.9970 | 0.9974 | 328ms | 5409ms | Excellent |
| K1V3-adaptive | 1 | 3 | adaptive | **8.9x** | 0.9925 | 0.9874 | 260ms | 4836ms | Good |
| K2V4-ANS | 2 | 4 | ANS | **5.9x** | 0.9968 | 0.9972 | 314ms | 5583ms | Excellent |
| K1V3-ANS | 1 | 3 | ANS | **8.9x** | 0.9924 | 0.9866 | 260ms | 4797ms | Good |
| K2V4-adapt+ANS | 2 | 4 | adaptive+ANS | **5.9x** | 0.9970 | 0.9974 | 313ms | 5427ms | Excellent |
| K1V3-adapt+ANS | 1 | 3 | adaptive+ANS | **8.9x** | 0.9925 | 0.9874 | 266ms | 4773ms | Good |
| K2V4-FULL | 2 | 4 | adaptive+ANS+fused | **5.9x** | 0.9970 | 0.9974 | 290ms | 5421ms | Excellent |
| K1V3-FULL | 1 | 3 | adaptive+ANS+fused | **8.9x** | 0.9925 | 0.9874 | 267ms | 4796ms | Good |
| K2V3-FULL | 2 | 3 | adaptive+ANS+fused | **7.2x** | 0.9970 | 0.9874 | 278ms | 5407ms | Good |
| K3V4-FULL | 3 | 4 | adaptive+ANS+fused | **5.0x** | 0.9993 | 0.9974 | 324ms | 5494ms | Excellent |
| K1V2-FULL | 1 | 2 | adaptive+ANS+fused | **12.8x** | 0.9925 | 0.9120 | 256ms | 4737ms | --- |
| K1V4-FULL | 1 | 4 | adaptive+ANS+fused | **7.1x** | 0.9925 | 0.9974 | 266ms | 4800ms | Excellent |

### Optimization Impact

**K2V4 baseline -> FULL:**
- Compression: 5.9x -> 5.9x (-0.1%)
- V Cosine: 0.9972 -> 0.9974 (+0.0002)
- Compress speed: 287ms -> 290ms (-1%)

**K1V3 baseline -> FULL:**
- Compression: 8.9x -> 8.9x (+0.1%)
- V Cosine: 0.9866 -> 0.9874 (+0.0008)


---
*Benchmarked 2026-03-31 18:28 by [@OnlyTerp](https://x.com/OnlyTerp) / Terp AI Labs*
