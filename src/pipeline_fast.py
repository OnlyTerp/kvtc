"""GPU-accelerated KVTC compression pipeline."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from .common import CalibrationData, CompressedKVCache, CompressedSection, CompressionMetadata
from .entropy import compress as entropy_compress
from .entropy import decompress as entropy_decompress
from .entropy import unpack_bits
from .gpu_ops import (
    batch_dequantize,
    batch_quantize,
    fast_pack_bits,
    fast_unpack_dequantize,
    greedy_bit_allocation,
    vectorized_quant_params,
)
from .pca import apply_rope, apply_rope_inverse, pca_inverse, pca_transform


class KVTCCompressorFast:
    """GPU-accelerated KVTC compressor. Drop-in replacement for KVTCCompressor."""

    def __init__(self, calibration_data: CalibrationData, device: str = "cpu") -> None:
        self.calibration_data = calibration_data
        self.device = device
        self._timing: Dict[str, float] = {}

    @property
    def timing(self) -> Dict[str, float]:
        return self._timing

    def compress(self, kv_cache: Dict[str, torch.Tensor], positions: torch.Tensor) -> CompressedKVCache:
        """Run the full KVTC pipeline with GPU acceleration."""
        self._timing = {}
        t_total = time.perf_counter()

        keys = kv_cache["keys"]
        values = kv_cache["values"]
        if keys.shape != values.shape:
            raise ValueError("Keys and values must have the same shape.")
        if keys.dim() != 4:
            raise ValueError("Expected KV tensors with shape [layers, tokens, heads, dim].")

        layers, tokens, heads, dim = keys.shape
        sink_len = min(self.calibration_data.sink_tokens, tokens)
        residual = max(tokens - sink_len, 0)
        window_len = min(self.calibration_data.window_tokens, residual)
        middle_len = max(tokens - sink_len - window_len, 0)
        middle_positions = positions[sink_len : sink_len + middle_len]

        sinks = {
            "keys": keys[:, :sink_len].clone(),
            "values": values[:, :sink_len].clone(),
        }
        window = {
            "keys": keys[:, tokens - window_len :].clone() if window_len else keys[:, 0:0].clone(),
            "values": values[:, tokens - window_len :].clone() if window_len else values[:, 0:0].clone(),
        }

        compressed_sections: List[CompressedSection] = []
        total_original_bytes = 0
        total_compressed_bytes = 0

        t_pca = 0.0
        t_quant = 0.0
        t_pack = 0.0
        t_dp = 0.0

        if middle_len > 0:
            for layer_idx in range(layers):
                for kind, tensor in (("keys", keys), ("values", values)):
                    middle = tensor[layer_idx, sink_len : sink_len + middle_len]
                    for group_idx, start in enumerate(range(0, heads, self.calibration_data.head_group_size)):
                        entry = self.calibration_data.entries[(layer_idx, group_idx, kind)]
                        group = middle[:, start : start + self.calibration_data.head_group_size, :]
                        group_heads = group.shape[1]

                        # RoPE undo + PCA transform
                        t0 = time.perf_counter()
                        work = group
                        if kind == "keys":
                            work = apply_rope_inverse(
                                work, middle_positions,
                                rope_theta=self.calibration_data.rope_theta,
                                head_dim=dim,
                            )
                        rows = work.reshape(middle_len * group_heads, dim).to(torch.float32)
                        ev = entry.eigenvectors.to(rows.device)
                        mn = entry.mean.to(rows.device)
                        centered = rows - mn
                        pca_values = pca_transform(centered, ev)
                        t_pca += time.perf_counter() - t0

                        # DP bit allocation (greedy — vectorized)
                        t0 = time.perf_counter()
                        bit_widths = greedy_bit_allocation(
                            entry.eigenvalues.to(rows.device),
                            entry.bit_budget,
                        )
                        t_dp += time.perf_counter() - t0

                        # Vectorized quantization
                        t0 = time.perf_counter()
                        params = vectorized_quant_params(pca_values, bit_widths)
                        indices = batch_quantize(
                            pca_values, bit_widths,
                            params.scales, params.zero_points,
                        )
                        t_quant += time.perf_counter() - t0

                        # Bit packing + entropy
                        t0 = time.perf_counter()
                        packed = fast_pack_bits(indices, bit_widths)
                        compressed_bytes, _ = entropy_compress(packed)
                        t_pack += time.perf_counter() - t0

                        total_original_bytes += rows.numel() * 2  # FP16
                        total_compressed_bytes += len(compressed_bytes)

                        compressed_sections.append(
                            CompressedSection(
                                layer_idx=layer_idx,
                                group_idx=group_idx,
                                kind=kind,
                                compressed_bytes=compressed_bytes,
                                packed_size=len(packed),
                                lengths=[rows.shape[0]] * dim,
                                num_rows=rows.shape[0],
                                group_heads=group_heads,
                                bit_widths=bit_widths.tolist(),
                                scales=params.scales.tolist(),
                                zero_points=params.zero_points.tolist(),
                                mins=params.mins.tolist(),
                            )
                        )

        ratio = total_original_bytes / max(total_compressed_bytes, 1) if total_original_bytes else 1.0
        metadata = CompressionMetadata(
            positions_middle=middle_positions.tolist(),
            original_shape=(layers, tokens, heads, dim),
            sink_len=sink_len,
            middle_len=middle_len,
            window_len=window_len,
            compression_ratio=ratio,
        )

        self._timing = {
            "pca_ms": t_pca * 1000,
            "dp_ms": t_dp * 1000,
            "quant_ms": t_quant * 1000,
            "pack_ms": t_pack * 1000,
            "total_ms": (time.perf_counter() - t_total) * 1000,
        }

        return CompressedKVCache(
            sinks=sinks, window=window,
            compressed_sections=compressed_sections,
            metadata=metadata,
        )

    def decompress(self, compressed_cache: CompressedKVCache) -> Dict[str, torch.Tensor]:
        """Reverse the full KVTC pipeline with GPU-accelerated dequantization."""
        t_total = time.perf_counter()

        if compressed_cache.metadata is None:
            raise ValueError("Missing metadata.")

        layers, tokens, heads, dim = compressed_cache.metadata.original_shape
        sink_len = compressed_cache.metadata.sink_len
        middle_len = compressed_cache.metadata.middle_len
        window_len = compressed_cache.metadata.window_len

        result = {
            "keys": torch.zeros((layers, tokens, heads, dim), dtype=compressed_cache.sinks["keys"].dtype),
            "values": torch.zeros((layers, tokens, heads, dim), dtype=compressed_cache.sinks["values"].dtype),
        }

        if sink_len:
            result["keys"][:, :sink_len] = compressed_cache.sinks["keys"]
            result["values"][:, :sink_len] = compressed_cache.sinks["values"]
        if window_len:
            result["keys"][:, tokens - window_len:] = compressed_cache.window["keys"]
            result["values"][:, tokens - window_len:] = compressed_cache.window["values"]

        positions_middle = torch.tensor(compressed_cache.metadata.positions_middle, dtype=torch.long)

        for section in compressed_cache.compressed_sections:
            entry = self.calibration_data.entries[(section.layer_idx, section.group_idx, section.kind)]

            bw = torch.tensor(section.bit_widths, dtype=torch.int64)
            scales = torch.tensor(section.scales, dtype=torch.float32)
            zero_points = torch.tensor(section.zero_points, dtype=torch.float32)

            # Unpack bits (original fast path)
            packed = entropy_decompress(section.compressed_bytes, section.packed_size)
            unpacked = unpack_bits(packed, section.bit_widths, section.lengths)
            
            # Stack into [num_rows, components] and batch dequantize
            indices = torch.stack(unpacked, dim=-1)
            dequantized = batch_dequantize(indices, bw, scales, zero_points)

            # PCA inverse + RoPE reapply
            ev = entry.eigenvectors.to(dequantized.device)
            mn = entry.mean.to(dequantized.device)
            restored = pca_inverse(dequantized, ev) + mn
            group = restored.reshape(middle_len, section.group_heads, dim)

            if section.kind == "keys":
                group = apply_rope(
                    group, positions_middle,
                    rope_theta=self.calibration_data.rope_theta,
                    head_dim=dim,
                )

            start = section.group_idx * self.calibration_data.head_group_size
            result[section.kind][section.layer_idx, sink_len : sink_len + middle_len, start : start + section.group_heads] = group.to(
                result[section.kind].dtype
            )

        self._timing["decompress_ms"] = (time.perf_counter() - t_total) * 1000
        return result
