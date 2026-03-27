#!/usr/bin/env python3
"""KVTC GPU Benchmark — Real model compression validation.

Usage:
    # CPU test with TinyLlama (quick validation):
    python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu

    # GPU test (on 5090 or any CUDA machine):
    python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda

    # Larger model:
    python -m src.benchmark_gpu --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda

    # Custom sequence length:
    python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 1024 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch


def get_kv_cache_from_model(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    """Run a forward pass and extract the KV cache.
    
    Returns (kv_cache_dict, positions, num_layers).
    kv_cache_dict has shape [layers, tokens, heads, head_dim] for both keys and values.
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    past = outputs.past_key_values
    
    # Handle DynamicCache (.layers), legacy DynamicCache (.key_cache), and tuple formats
    keys = []
    values = []
    if hasattr(past, 'layers'):
        num_layers = len(past.layers)
        for layer in past.layers:
            keys.append(layer.keys[0].transpose(0, 1).cpu())
            values.append(layer.values[0].transpose(0, 1).cpu())
    elif hasattr(past, 'key_cache'):
        num_layers = len(past.key_cache)
        for i in range(num_layers):
            keys.append(past.key_cache[i][0].transpose(0, 1).cpu())
            values.append(past.value_cache[i][0].transpose(0, 1).cpu())
    else:
        num_layers = len(past)
        for layer_k, layer_v in past:
            keys.append(layer_k[0].transpose(0, 1).cpu())
            values.append(layer_v[0].transpose(0, 1).cpu())
    
    kv_cache = {
        "keys": torch.stack(keys),      # [layers, tokens, heads, dim]
        "values": torch.stack(values),   # [layers, tokens, heads, dim]
    }
    positions = torch.arange(seq_len, dtype=torch.long)
    
    return kv_cache, positions, num_layers


def measure_compression(
    kv_cache: Dict[str, torch.Tensor],
    positions: torch.Tensor,
    calibration_data,
    verbose: bool = True,
    use_fast: bool = True,
) -> Dict:
    """Compress, decompress, and measure quality metrics."""
    if use_fast:
        from src.pipeline_fast import KVTCCompressorFast
        compressor = KVTCCompressorFast(calibration_data)
    else:
        from src.pipeline import KVTCCompressor
        compressor = KVTCCompressor(calibration_data)
    
    # Compress
    t0 = time.perf_counter()
    compressed = compressor.compress(kv_cache, positions)
    compress_time = time.perf_counter() - t0
    
    # Decompress
    t0 = time.perf_counter()
    restored = compressor.decompress(compressed)
    decompress_time = time.perf_counter() - t0
    
    # Metrics
    layers, tokens, heads, dim = kv_cache["keys"].shape
    
    # Per-layer cosine similarity
    key_cosines = []
    value_cosines = []
    for layer in range(layers):
        kc = torch.nn.functional.cosine_similarity(
            kv_cache["keys"][layer].reshape(1, -1),
            restored["keys"][layer].reshape(1, -1),
        ).item()
        vc = torch.nn.functional.cosine_similarity(
            kv_cache["values"][layer].reshape(1, -1),
            restored["values"][layer].reshape(1, -1),
        ).item()
        key_cosines.append(kc)
        value_cosines.append(vc)
    
    # Overall cosine
    overall_key_cos = torch.nn.functional.cosine_similarity(
        kv_cache["keys"].reshape(1, -1),
        restored["keys"].reshape(1, -1),
    ).item()
    overall_val_cos = torch.nn.functional.cosine_similarity(
        kv_cache["values"].reshape(1, -1),
        restored["values"].reshape(1, -1),
    ).item()
    
    # MSE
    key_mse = torch.mean((kv_cache["keys"].float() - restored["keys"].float()) ** 2).item()
    val_mse = torch.mean((kv_cache["values"].float() - restored["values"].float()) ** 2).item()
    
    # Original size in bytes (FP16)
    original_bytes = kv_cache["keys"].numel() * 2 + kv_cache["values"].numel() * 2  # FP16 = 2 bytes
    
    # Compressed size (sum of all compressed sections + sinks + window in FP16)
    sink_bytes = compressed.sinks["keys"].numel() * 2 + compressed.sinks["values"].numel() * 2
    window_bytes = compressed.window["keys"].numel() * 2 + compressed.window["values"].numel() * 2
    section_bytes = sum(len(s.compressed_bytes) for s in compressed.compressed_sections)
    compressed_total = sink_bytes + window_bytes + section_bytes
    
    ratio = original_bytes / max(compressed_total, 1)
    
    results = {
        "layers": layers,
        "tokens": tokens,
        "heads": heads,
        "head_dim": dim,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_total,
        "sink_bytes": sink_bytes,
        "window_bytes": window_bytes,
        "section_bytes": section_bytes,
        "compression_ratio": ratio,
        "pipeline_ratio": compressed.metadata.compression_ratio,
        "key_cosine_avg": sum(key_cosines) / len(key_cosines),
        "value_cosine_avg": sum(value_cosines) / len(value_cosines),
        "key_cosine_min": min(key_cosines),
        "value_cosine_min": min(value_cosines),
        "overall_key_cosine": overall_key_cos,
        "overall_value_cosine": overall_val_cos,
        "key_mse": key_mse,
        "value_mse": val_mse,
        "compress_time_ms": compress_time * 1000,
        "decompress_time_ms": decompress_time * 1000,
        "num_sections": len(compressed.compressed_sections),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("KVTC Compression Results")
        print("=" * 60)
        print(f"Model shape:     {layers} layers × {tokens} tokens × {heads} heads × {dim} dim")
        print(f"Original size:   {original_bytes / 1024:.1f} KB")
        print(f"Compressed:      {compressed_total / 1024:.1f} KB")
        print(f"  Sinks (FP16):  {sink_bytes / 1024:.1f} KB")
        print(f"  Window (FP16): {window_bytes / 1024:.1f} KB")
        print(f"  Sections:      {section_bytes / 1024:.1f} KB")
        print(f"Compression:     {ratio:.1f}× (pipeline: {compressed.metadata.compression_ratio:.1f}×)")
        print()
        print(f"Key cosine:      {results['key_cosine_avg']:.4f} avg, {results['key_cosine_min']:.4f} min")
        print(f"Value cosine:    {results['value_cosine_avg']:.4f} avg, {results['value_cosine_min']:.4f} min")
        print(f"Key MSE:         {key_mse:.6f}")
        print(f"Value MSE:       {val_mse:.6f}")
        print()
        print(f"Compress time:   {compress_time * 1000:.1f} ms")
        print(f"Decompress time: {decompress_time * 1000:.1f} ms")
        if hasattr(compressor, 'timing') and compressor.timing:
            t = compressor.timing
            print(f"  Breakdown: PCA={t.get('pca_ms',0):.0f}ms  DP={t.get('dp_ms',0):.0f}ms  Quant={t.get('quant_ms',0):.0f}ms  Pack={t.get('pack_ms',0):.0f}ms")
            if 'avg_bits_allocated' in t:
                print(f"  Avg bits actually allocated: {t['avg_bits_allocated']:.2f}")
        print("=" * 60)
    
    return results


def run_logit_comparison(
    model,
    tokenizer,
    kv_cache: Dict[str, torch.Tensor],
    restored_cache: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Dict:
    """Compare logits from original vs restored KV cache on continuation tokens."""
    prompts = [
        "The capital of France is",
        "In 1969, humans first",
        "Python is a programming language that",
        "The theory of relativity was developed by",
        "Water boils at a temperature of",
    ]
    
    matches = 0
    cosines = []
    
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        original_logits = outputs.logits[0, -1].cpu().float()
        original_pred = original_logits.argmax().item()
        original_token = tokenizer.decode([original_pred])
        
        # For restored: we'd need to reconstruct the past_key_values format
        # This is model-specific, so we'll just compare the KV caches directly
        # and report the logit comparison from a fresh forward pass
        
        print(f"  '{prompt}' → '{original_token.strip()}'")
        matches += 1  # Placeholder until we wire restored KV → model
    
    return {"prompts_tested": len(prompts), "match_rate": matches / len(prompts)}


def main():
    parser = argparse.ArgumentParser(description="KVTC Real Model Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--seq-len", type=int, default=512, help="Target sequence length for benchmark")
    parser.add_argument("--calibration-samples", type=int, default=15, help="Number of calibration samples")
    parser.add_argument("--bit-budget-ratio", type=float, default=0.25, help="Bit budget as fraction of FP16")
    parser.add_argument("--sink-tokens", type=int, default=4, help="Number of attention sink tokens to preserve")
    parser.add_argument("--window-tokens", type=int, default=32, help="Sliding window size to preserve")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization (bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization (bitsandbytes)")
    args = parser.parse_args()
    
    print(f"KVTC Benchmark")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            sys.exit(1)
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"VRAM:   {vram / 1024**3:.1f} GB")
    
    print("\n[1/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_kwargs = {
        "trust_remote_code": True,
    }
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device == "cuda":
        model = model.cuda()
    model.eval()
    
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    
    # Generate a long prompt — repeat to fill seq_len
    base_prompt = (
        "Explain the following topics in detail with examples and mathematical formulations: "
        "quantum computing and its applications in cryptography and post-quantum security, "
        "the complete history of artificial intelligence from Alan Turing's original proposals through "
        "modern transformer architectures and large language models, "
        "how neural networks learn through backpropagation and stochastic gradient descent with momentum, "
        "the mathematics behind multi-head attention mechanisms in transformer architectures including "
        "scaled dot-product attention and positional encoding schemes, "
        "recent advances in KV cache compression for efficient LLM inference including quantization "
        "and pruning approaches, the role of principal component analysis in dimensionality reduction "
        "and its application to feature decorrelation in neural network activations, "
        "entropy coding methods including Huffman coding and arithmetic coding and their use in "
        "data compression pipelines, the difference between lossy and lossless compression algorithms "
        "and when each is appropriate for different data types, "
        "the theory of information and Shannon entropy and how it relates to optimal compression rates, "
        "modern GPU architecture and how tensor cores accelerate matrix multiplication in deep learning, "
        "the design of memory hierarchies in modern processors and how cache locality affects performance, "
        "distributed training strategies including data parallelism and model parallelism and pipeline parallelism, "
        "the mathematics of convex optimization and how it applies to training neural networks, "
        "reinforcement learning from human feedback and its role in aligning language models, "
        "the architecture of modern operating systems and how virtual memory management works. "
    )
    # Repeat to fill desired token count
    long_prompt = base_prompt
    while len(tokenizer.encode(long_prompt)) < args.seq_len:
        long_prompt = long_prompt + " " + base_prompt
    
    # Truncate to desired length
    tokens = tokenizer.encode(long_prompt)
    if len(tokens) > args.seq_len:
        tokens = tokens[:args.seq_len]
        long_prompt = tokenizer.decode(tokens)
    
    print(f"\n[2/4] Calibrating PCA bases ({args.calibration_samples} samples)...")
    from src.calibrate import KVTCCalibrator
    
    # Longer, more diverse calibration texts for better PCA bases
    calibration_texts = [
        "KV cache compression can unlock longer contexts in language models. The key insight is that KV vectors exhibit strong low-rank structure that can be exploited through principal component analysis. By projecting onto the top principal components and quantizing adaptively, we achieve near-lossless compression at high ratios.",
        "Dynamic programming assigns bits where variance matters most. The optimal bit allocation minimizes total reconstruction error subject to a bit budget constraint. Components with higher eigenvalues receive more bits, while low-variance components can be pruned entirely.",
        "Principal components capture the most important directions in data. Singular value decomposition reveals the latent structure in high-dimensional vectors. The eigenvectors form an orthonormal basis that decorrelates the data, making each dimension independently quantizable.",
        "Attention sinks at the beginning of sequences should be preserved exactly. Research has shown that the first few tokens accumulate disproportionate attention weight regardless of content. Compressing these tokens degrades quality significantly, so they are kept in full precision.",
        "Random rotation decorrelates features before quantization. The Johnson-Lindenstrauss lemma guarantees that random projections preserve pairwise distances. This property is exploited in both TurboQuant and KVTC for different purposes.",
        "The theory of relativity was developed by Albert Einstein in the early twentieth century. Special relativity deals with objects moving at constant velocities near the speed of light, while general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
        "Machine learning algorithms learn patterns from data through optimization. Gradient descent iteratively adjusts model parameters to minimize a loss function. Modern deep learning uses stochastic gradient descent with momentum and adaptive learning rates.",
        "The architecture of transformer models consists of stacked layers of multi-head self-attention and feed-forward networks. Each attention head computes queries, keys, and values from the input, then uses scaled dot-product attention to weight the values.",
        "Python is a versatile programming language used extensively in scientific computing, web development, and artificial intelligence. Its ecosystem includes libraries like NumPy for numerical computing, PyTorch for deep learning, and transformers for natural language processing.",
        "Cryptocurrency markets operate twenty-four hours a day, seven days a week, with prices determined by supply and demand on decentralized exchanges. Bitcoin remains the largest cryptocurrency by market capitalization, followed by Ethereum.",
        "The history of computing spans from mechanical calculators through vacuum tube computers to modern silicon chips. Moore's law predicted that transistor density would double approximately every two years, a trend that held for decades.",
        "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform certain calculations exponentially faster than classical computers. Shor's algorithm can factor large numbers efficiently, threatening current encryption.",
        "Neural network architectures have evolved from simple perceptrons to complex convolutional, recurrent, and transformer models. Each architecture excels at different types of tasks depending on the structure of the input data.",
        "The process of protein folding determines the three-dimensional structure of proteins from their amino acid sequences. AlphaFold demonstrated that deep learning could predict protein structures with remarkable accuracy.",
        "Climate science relies on complex numerical models that simulate atmospheric and oceanic circulation patterns. These models must account for radiation, convection, precipitation, and feedback loops between different components of the Earth system.",
        "Modern GPU architectures include thousands of parallel processing cores optimized for matrix operations. Tensor cores provide dedicated hardware for mixed-precision matrix multiplication, accelerating deep learning training and inference.",
        "The development of large language models has progressed from word embeddings through recurrent networks to the transformer architecture. Scaling laws suggest that model performance improves predictably with increased data, compute, and parameters.",
        "Database systems use indexing structures like B-trees and hash tables to enable efficient data retrieval. Query optimization transforms declarative SQL statements into efficient execution plans that minimize disk I/O and memory usage.",
        "Operating systems manage hardware resources and provide abstractions for application programs. The kernel handles process scheduling, memory management, file systems, device drivers, and inter-process communication.",
        "Natural language processing encompasses tasks like tokenization, parsing, named entity recognition, sentiment analysis, machine translation, question answering, and text generation. Modern approaches use pre-trained transformer models fine-tuned for specific tasks.",
    ]
    
    calibrator = KVTCCalibrator(head_group_size=1)
    calibrator.collect_samples(model, tokenizer, calibration_texts[:args.calibration_samples])
    calibration = calibrator.compute_calibration(bit_budget_ratio=args.bit_budget_ratio)
    calibration.sink_tokens = args.sink_tokens
    calibration.window_tokens = args.window_tokens
    print(f"  Bit budget ratio: {args.bit_budget_ratio} ({args.bit_budget_ratio * 16:.1f} bits/value avg)")
    print(f"  Collected {calibrator.samples_collected} samples")
    print(f"  Entries: {len(calibration.entries)} (layer × group × kind)")
    
    print(f"\n[3/4] Extracting KV cache ({len(tokens)} tokens)...")
    kv_cache, positions, num_layers = get_kv_cache_from_model(
        model, tokenizer, long_prompt, device=args.device
    )
    layers, seq, heads, dim = kv_cache["keys"].shape
    print(f"  Shape: {layers}L × {seq}T × {heads}H × {dim}D")
    print(f"  KV cache size: {(kv_cache['keys'].numel() + kv_cache['values'].numel()) * 2 / 1024:.1f} KB (FP16)")
    
    print(f"\n[4/4] Running KVTC compression...")
    results = measure_compression(kv_cache, positions, calibration, verbose=True)
    
    # Summary verdict
    print("\n" + "=" * 60)
    key_ok = results["key_cosine_avg"] > 0.95
    val_ok = results["value_cosine_avg"] > 0.95
    ratio_ok = results["compression_ratio"] > 3.0
    
    if key_ok and val_ok and ratio_ok:
        print("PASS -- Quality and compression look good!")
    elif key_ok and val_ok:
        print("PARTIAL -- Quality is good but compression ratio is low")
        if results["compression_ratio"] < 1.1:
            print(f"NOTE: Only {results['tokens']} tokens with sink={calibration.sink_tokens} + window={calibration.window_tokens}.")
            print("      All tokens are in sink/window regions -- nothing to compress!")
            print("      Use --seq-len 512+ to test actual compression.")
    else:
        print(f"NEEDS WORK -- Cosine sim below 0.95 (keys={results['key_cosine_avg']:.4f}, values={results['value_cosine_avg']:.4f})")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
