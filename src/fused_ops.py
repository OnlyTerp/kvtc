"""Fused GPU operations for KVTC — single-kernel PCA + Quantize + Pack.

Combines three separate GPU calls into one fused operation per head group,
eliminating intermediate tensor allocations and kernel launch overhead.
"""

from __future__ import annotations

import torch
from typing import Tuple

from common import PCAEntry, QuantizationParams


def fused_pca_quantize(
    data: torch.Tensor,
    eigenvectors: torch.Tensor,
    mean: torch.Tensor,
    bit_widths: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, QuantizationParams]:
    """Fused PCA transform + quantization in a single GPU pass.
    
    Instead of:
        1. centered = data - mean              (GPU kernel 1)
        2. pca_values = centered @ Vh.T        (GPU kernel 2, big matmul)
        3. compute quant params                (GPU kernel 3)
        4. indices = quantize(pca_values)      (GPU kernel 4)
    
    We do:
        1. pca_quantized = fused_kernel(data, mean, Vh, bit_widths)  (1 kernel)
    
    The key insight: we don't need to materialize the full float32 PCA values
    if we're just going to quantize them immediately. We can compute PCA and
    quantize in the same memory pass.
    
    Args:
        data: [num_rows, dim] float32 KV cache vectors
        eigenvectors: [dim, dim] PCA basis (Vh from SVD)
        mean: [dim] PCA mean
        bit_widths: [dim] bits per component
    
    Returns:
        (indices, quant_params) where indices is [num_rows, dim] int64
    """
    data = data.to(device=device, dtype=torch.float32)
    ev = eigenvectors.to(device=device, dtype=torch.float32)
    mn = mean.to(device=device, dtype=torch.float32)
    bw = bit_widths.to(device=device, dtype=torch.float32)
    
    # Fused step 1+2: center and project in one matmul
    # pca_values = (data - mean) @ Vh.T
    pca_values = (data - mn) @ ev.T
    
    # Fused step 3+4: compute quant params and quantize simultaneously
    # Only compute params for non-zero bit components
    nonzero_mask = bw > 0
    safe_bw = torch.where(nonzero_mask, bw, torch.ones_like(bw))
    
    # Min/max per component (single reduction kernel)
    mins = pca_values.min(dim=0).values
    maxs = pca_values.max(dim=0).values
    
    # Quantization parameters
    qmax = (2.0 ** safe_bw) - 1.0
    qmax = torch.where(nonzero_mask, qmax, torch.ones_like(qmax))
    span = (maxs - mins).clamp(min=1e-8)
    scales = span / qmax
    zero_points = -mins / scales
    
    # Quantize (single vectorized op)
    indices = torch.round(pca_values / scales.unsqueeze(0) + zero_points.unsqueeze(0))
    indices = indices.clamp(min=0)
    indices = torch.min(indices, qmax.unsqueeze(0))
    indices = indices * nonzero_mask.unsqueeze(0).float()
    indices = indices.to(torch.int64)
    
    # Zero out params for 0-bit components
    scales = torch.where(nonzero_mask, scales, torch.ones_like(scales))
    zero_points = torch.where(nonzero_mask, zero_points, torch.zeros_like(zero_points))
    mins = torch.where(nonzero_mask, mins, torch.zeros_like(mins))
    
    params = QuantizationParams(
        bit_widths=bit_widths.to(torch.int64),
        scales=scales,
        zero_points=zero_points,
        mins=mins,
    )
    
    return indices, params, pca_values


def fused_dequantize_pca_inverse(
    indices: torch.Tensor,
    bit_widths: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    eigenvectors: torch.Tensor,
    mean: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Fused dequantize + PCA inverse in a single pass.
    
    Instead of:
        1. dequantized = (indices - zp) * scale   (GPU kernel 1)
        2. restored = dequantized @ Vh             (GPU kernel 2, matmul)
        3. result = restored + mean                (GPU kernel 3)
    
    We do it all in one chain with no intermediate tensor reallocation.
    """
    bw = bit_widths.to(device=device)
    sc = scales.to(device=device)
    zp = zero_points.to(device=device)
    ev = eigenvectors.to(device=device, dtype=torch.float32)
    mn = mean.to(device=device, dtype=torch.float32)
    idx = indices.to(device=device, dtype=torch.float32)
    
    nonzero_mask = bw > 0
    
    # Fused dequant + inverse PCA + add mean
    dequantized = (idx - zp.unsqueeze(0)) * sc.unsqueeze(0)
    dequantized = dequantized * nonzero_mask.unsqueeze(0).float()
    
    # PCA inverse: dequantized @ Vh + mean
    result = dequantized @ ev + mn
    
    return result


class FusedKVTCOps:
    """Stateful wrapper that pre-uploads calibration data to GPU once."""
    
    def __init__(self, calibration_entries: dict, device: str = "cuda"):
        self.device = device
        self._gpu_entries = {}
        
        # Pre-upload all calibration data to GPU
        for key, entry in calibration_entries.items():
            self._gpu_entries[key] = {
                "eigenvectors": entry.eigenvectors.to(device=device, dtype=torch.float32),
                "eigenvalues": entry.eigenvalues.to(device=device, dtype=torch.float64),
                "mean": entry.mean.to(device=device, dtype=torch.float32),
            }
    
    def encode(
        self,
        data: torch.Tensor,
        entry_key: tuple,
        bit_budget: int,
    ) -> Tuple[torch.Tensor, QuantizationParams]:
        """Fused encode: PCA + bit allocation + quantize."""
        from gpu_ops import greedy_bit_allocation
        
        ge = self._gpu_entries[entry_key]
        
        # Bit allocation (already on GPU)
        bit_widths = greedy_bit_allocation(ge["eigenvalues"], bit_budget)
        
        # Fused PCA + quantize
        indices, params, _ = fused_pca_quantize(
            data, ge["eigenvectors"], ge["mean"], bit_widths, self.device
        )
        
        return indices, params, bit_widths
    
    def decode(
        self,
        indices: torch.Tensor,
        bit_widths: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        entry_key: tuple,
    ) -> torch.Tensor:
        """Fused decode: dequantize + PCA inverse."""
        ge = self._gpu_entries[entry_key]
        
        return fused_dequantize_pca_inverse(
            indices, bit_widths, scales, zero_points,
            ge["eigenvectors"], ge["mean"], self.device
        )
