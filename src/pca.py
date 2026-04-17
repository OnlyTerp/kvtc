"""PCA and RoPE utilities for KVTC."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import math

from .common import CalibrationData, CalibrationKey, PCAEntry
from .quantize import dp_bit_allocation


def pca_transform(centered: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Project centered data into PCA space.
    
    Args:
        centered: [num_rows, dim] centered data (mean already subtracted)
        eigenvectors: [dim, dim] or [k, dim] PCA basis (rows are eigenvectors)
    
    Returns:
        [num_rows, k] PCA coefficients
    """
    # eigenvectors from SVD: Vh has shape [k, dim], each row is an eigenvector
    # projection = centered @ Vh.T
    return centered @ eigenvectors.T


def pca_inverse(pca_values: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Reconstruct from PCA space back to original space.
    
    Args:
        pca_values: [num_rows, k] PCA coefficients
        eigenvectors: [k, dim] PCA basis
    
    Returns:
        [num_rows, dim] reconstructed data (without mean)
    """
    return pca_values @ eigenvectors


def _rotary_embedding(positions: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Compute rotary position embeddings (cos, sin).
    
    Args:
        positions: [seq_len] position indices
        dim: head dimension
        theta: RoPE base frequency
    
    Returns:
        (cos, sin) each of shape [seq_len, dim//2]
    """
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=positions.device) / half_dim))
    # [seq_len, half_dim]
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x: [seq_len, heads, dim] or [seq_len, dim]
        cos: [seq_len, dim//2]
        sin: [seq_len, dim//2]
    
    Returns:
        Tensor with same shape as x, with RoPE applied
    """
    if x.dim() == 3:
        # [seq_len, heads, dim]
        seq_len, heads, dim = x.shape
        half_dim = dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        # Broadcast cos/sin: [seq_len, 1, half_dim]
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        out1 = x1 * cos_b - x2 * sin_b
        out2 = x2 * cos_b + x1 * sin_b
        return torch.cat([out1, out2], dim=-1)
    elif x.dim() == 2:
        # [seq_len, dim]
        dim = x.shape[-1]
        half_dim = dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    head_dim: int = 128,
) -> torch.Tensor:
    """Apply RoPE rotation to key/value tensor.
    
    Args:
        x: [seq_len, heads, dim] tensor
        positions: [seq_len] position indices
        rope_theta: RoPE base frequency
        head_dim: dimension of each head
    
    Returns:
        Tensor with RoPE applied
    
    Raises:
        ValueError: If head_dim is odd (RoPE requires even dimensions).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    cos, sin = _rotary_embedding(positions, head_dim, rope_theta)
    cos = cos.to(x.dtype).to(x.device)
    sin = sin.to(x.dtype).to(x.device)
    return _apply_rotary_emb(x, cos, sin)


def apply_rope_inverse(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    head_dim: int = 128,
) -> torch.Tensor:
    """Undo RoPE rotation (apply inverse rotation).
    
    RoPE inverse is just applying with negated sin (rotation in opposite direction).
    
    Args:
        x: [seq_len, heads, dim] tensor with RoPE already applied
        positions: [seq_len] position indices
        rope_theta: RoPE base frequency
        head_dim: dimension of each head
    
    Returns:
        Tensor with RoPE undone
    """
    cos, sin = _rotary_embedding(positions, head_dim, rope_theta)
    cos = cos.to(x.dtype).to(x.device)
    sin = sin.to(x.dtype).to(x.device)
    # Inverse rotation: negate sin
    return _apply_rotary_emb(x, cos, -sin)


# ---------------------------------------------------------------------------
# PCACalibrator — collects KV samples and computes PCA calibration data
# ---------------------------------------------------------------------------


@dataclass
class PCACalibrator:
    """Collect KV cache samples and compute PCA calibration artifacts.

    Usage:
        calibrator = PCACalibrator(head_group_size=2)
        for layer_idx in range(num_layers):
            calibrator.collect(layer_idx, "keys", key_tensor, positions)
            calibrator.collect(layer_idx, "values", value_tensor)
        calibration = calibrator.compute(bit_budget_ratio=0.25)
    """

    head_group_size: int = 1
    rope_theta: float = 10000.0

    # Internal accumulators: {(layer, group, kind): [list of tensors]}
    _samples: Dict[CalibrationKey, List[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _positions: Dict[int, List[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def collect(
        self,
        layer_idx: int,
        kind: str,
        tensor: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate KV cache samples for one layer/kind.

        Args:
            layer_idx: Transformer layer index.
            kind: ``"keys"`` or ``"values"``.
            tensor: ``[seq_len, heads, dim]`` KV tensor (already detached, on CPU).
            positions: ``[seq_len]`` position indices (required for keys).
        """
        if tensor.numel() == 0:
            return

        # tensor shape: [seq_len, heads, dim]
        seq_len = tensor.shape[0]
        if tensor.dim() == 2:
            # [seq_len, dim] — single head
            tensor = tensor.unsqueeze(1)
        heads = tensor.shape[1]
        dim = tensor.shape[2]

        # Undo RoPE for keys so PCA sees the underlying low-rank structure
        if kind == "keys" and positions is not None:
            tensor = apply_rope_inverse(tensor, positions, rope_theta=self.rope_theta, head_dim=dim)

        # Group heads
        for group_start in range(0, heads, self.head_group_size):
            group_end = min(group_start + self.head_group_size, heads)
            group_idx = group_start // self.head_group_size
            key: CalibrationKey = (layer_idx, group_idx, kind)
            # Flatten group heads into rows: [seq_len * group_heads, dim]
            group_data = tensor[:, group_start:group_end, :].reshape(-1, dim).float()
            self._samples[key].append(group_data)

        if positions is not None:
            self._positions[layer_idx].append(positions)

    def compute(self, bit_budget_ratio: float = 0.25) -> CalibrationData:
        """Compute PCA bases and bit allocations from accumulated samples.

        Args:
            bit_budget_ratio: Fraction of original bits to allocate (0-1).
                              E.g. 0.25 means 4 bits per 16-bit component on average.

        Returns:
            :class:`CalibrationData` ready for the compressor.
        """
        entries: Dict[CalibrationKey, PCAEntry] = {}

        for key, sample_list in self._samples.items():
            layer_idx, group_idx, kind = key

            if not sample_list:
                continue

            # Concatenate all collected samples
            data = torch.cat(sample_list, dim=0)  # [total_rows, dim]
            dim = data.shape[1]

            # Compute PCA via SVD
            mean = data.mean(dim=0)
            centered = data - mean
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            # eigenvalues from singular values
            eigenvalues = singular_values.square() / max(centered.shape[0] - 1, 1)

            # vh has shape [k, dim] where k = min(num_rows, dim).
            # When k < dim, pad with zero-eigenvalue directions to get [dim, dim].
            k = vh.shape[0]
            if k < dim:
                pad = torch.zeros(dim - k, dim, dtype=vh.dtype, device=vh.device)
                vh_full = torch.cat([vh, pad], dim=0)
                eigenvalues = torch.cat(
                    [eigenvalues, torch.zeros(dim - k, dtype=eigenvalues.dtype, device=eigenvalues.device)]
                )
            else:
                vh_full = vh

            # eigenvectors: [dim, dim] — rows are eigenvectors (same convention as SVD vh)
            eigenvectors = vh_full

            # Determine bit budget
            total_bits = int(dim * 16 * bit_budget_ratio)
            bit_widths = dp_bit_allocation(eigenvalues, bit_budget=total_bits, max_bits=16)
            bit_budget = int(bit_widths.sum().item())

            # Compute per-component ranges for quantization
            pca_coeffs = pca_transform(centered, eigenvectors)
            pca_mins = pca_coeffs.min(dim=0).values
            pca_maxs = pca_coeffs.max(dim=0).values

            # Build head indices for this group
            group_start = group_idx * self.head_group_size
            head_indices = list(range(group_start, group_start + self.head_group_size))

            entries[key] = PCAEntry(
                eigenvectors=eigenvectors,  # [dim, dim] — rows are eigenvectors
                eigenvalues=eigenvalues,
                mean=mean,
                head_indices=head_indices,
                kind=kind,
                bit_budget=bit_budget,
                pca_mins=pca_mins,
                pca_maxs=pca_maxs,
                bit_widths=bit_widths,
            )

        return CalibrationData(
            entries=entries,
            head_group_size=self.head_group_size,
            rope_theta=self.rope_theta,
        )
