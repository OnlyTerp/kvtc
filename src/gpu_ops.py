"""GPU-optimized KVTC operations — vectorized PyTorch, no Python loops."""

from __future__ import annotations

import torch

from .common import QuantizationParams


def greedy_bit_allocation(
    eigenvalues: torch.Tensor,
    bit_budget: int,
    max_bits: int = 16,
) -> torch.Tensor:
    """Vectorized greedy bit allocation via precomputed gain sorting.
    
    Minimizes sum(λᵢ / 4^bᵢ) subject to sum(bᵢ) ≤ B.
    
    gain(i, b) = λᵢ × 3 / 4^b = marginal MSE reduction from assigning
    the b-th bit to component i. We precompute ALL gains, sort them,
    and greedily take the top B. Each entry (i, b) is only valid if
    all entries (i, 1), (i, 2), ..., (i, b-1) are also taken.
    Since gains are monotonically decreasing in b for each i, the
    sorted order naturally respects this constraint.
    """
    d = eigenvalues.numel()
    ev = eigenvalues.detach().to(torch.float64).flatten()
    budget = max(int(bit_budget), 0)
    
    if budget == 0 or d == 0:
        return torch.zeros(d, dtype=torch.int64)
    
    # gains_matrix[i, b-1] = λᵢ × 3 / 4^b (gain of b-th bit on component i)
    bit_levels = torch.arange(1, max_bits + 1, dtype=torch.float64)  # [max_bits]
    gains_matrix = ev.unsqueeze(1) * 3.0 / (4.0 ** bit_levels.unsqueeze(0))  # [d, max_bits]
    
    flat_gains = gains_matrix.flatten()  # [d * max_bits]
    total_slots = flat_gains.numel()
    
    if budget >= total_slots:
        return torch.full((d,), max_bits, dtype=torch.int64)
    
    # Take the top `budget` gains
    _, top_indices = torch.topk(flat_gains, min(budget, total_slots))
    
    # Each index maps to (component, bit_level):
    # index = component * max_bits + (bit_level - 1)
    # For each component, the NUMBER of its entries in top-B = its bit allocation
    # This works because gains are monotonically decreasing in bit_level,
    # so if the 3rd bit of component i is taken, bits 1 and 2 must also be taken.
    comp_indices = top_indices // max_bits  # [budget]
    
    # Count occurrences per component using bincount
    bits = torch.bincount(comp_indices, minlength=d).to(torch.int64)[:d]
    bits = bits.clamp(max=max_bits)
    
    return bits


def vectorized_quant_params(
    pca_values: torch.Tensor,
    bit_widths: torch.Tensor,
) -> QuantizationParams:
    """Compute quantization parameters without Python loops."""
    if pca_values.dim() != 2:
        raise ValueError("Expected pca_values shape [num_rows, components].")
    
    d = pca_values.shape[1]
    mins = pca_values.min(dim=0).values.to(torch.float32)
    maxs = pca_values.max(dim=0).values.to(torch.float32)
    bw = bit_widths.to(torch.float32)
    
    # Compute qmax per component: (1 << bits) - 1, but handle 0-bit
    # For 0-bit components: qmax=1 (avoid div by zero), then zero out later
    nonzero_mask = bw > 0
    safe_bw = torch.where(nonzero_mask, bw, torch.ones_like(bw))
    qmax = (2.0 ** safe_bw) - 1.0
    qmax = torch.where(nonzero_mask, qmax, torch.ones_like(qmax))
    
    span = (maxs - mins).clamp(min=1e-8)
    scales = span / qmax
    zero_points = -mins / scales
    
    # Zero out 0-bit components
    scales = torch.where(nonzero_mask, scales, torch.ones_like(scales))
    zero_points = torch.where(nonzero_mask, zero_points, torch.zeros_like(zero_points))
    mins = torch.where(nonzero_mask, mins, torch.zeros_like(mins))
    
    return QuantizationParams(
        bit_widths=bit_widths.to(torch.int64),
        scales=scales,
        zero_points=zero_points,
        mins=mins,
    )


def batch_quantize(
    pca_values: torch.Tensor,
    bit_widths: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Quantize all components at once — single tensor op.
    
    Args:
        pca_values: [num_rows, components] float tensor
        bit_widths: [components] int tensor
        scales: [components] float tensor
        zero_points: [components] float tensor
    
    Returns:
        indices: [num_rows, components] int64 tensor
    """
    bw = bit_widths.to(torch.float32)
    nonzero_mask = bw > 0  # [components]
    safe_bw = torch.where(nonzero_mask, bw, torch.ones_like(bw))
    qmax = (2.0 ** safe_bw - 1.0).unsqueeze(0)  # [1, components]
    
    # Quantize: round(value / scale + zero_point), clamp to [0, qmax]
    s = scales.unsqueeze(0)  # [1, components]
    zp = zero_points.unsqueeze(0)  # [1, components]
    
    indices = torch.round(pca_values / s + zp)
    indices = indices.clamp(min=0)
    indices = torch.min(indices, qmax)  # per-component clamp
    
    # Zero out 0-bit components
    indices = indices * nonzero_mask.unsqueeze(0).float()
    
    return indices.to(torch.int64)


def batch_dequantize(
    indices: torch.Tensor,
    bit_widths: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Dequantize all components at once — single tensor op.
    
    Args:
        indices: [num_rows, components] int64 tensor
        bit_widths: [components] int tensor
        scales: [components] float tensor
        zero_points: [components] float tensor
    
    Returns:
        values: [num_rows, components] float32 tensor
    """
    nonzero_mask = bit_widths > 0  # [components]
    
    s = scales.unsqueeze(0)  # [1, components]
    zp = zero_points.unsqueeze(0)  # [1, components]
    
    values = (indices.to(torch.float32) - zp) * s
    
    # Zero out 0-bit components
    values = values * nonzero_mask.unsqueeze(0).float()
    
    return values


def fast_pack_bits(
    indices: torch.Tensor,
    bit_widths: torch.Tensor,
) -> bytes:
    """Pack variable-width quantized indices into bytes.
    
    Uses GPU-accelerated packing if available, falls back to numpy/python.
    """
    try:
        from .triton_kernels import gpu_pack_variable_width, HAS_TRITON
        if HAS_TRITON or True:  # gpu_pack uses torch ops, doesn't need triton
            return gpu_pack_variable_width(indices, bit_widths)
    except ImportError:
        pass
    try:
        import numpy as np
        return _pack_bits_numpy(indices, bit_widths)
    except ImportError:
        return _pack_bits_python(indices, bit_widths)


def _pack_bits_numpy(indices: torch.Tensor, bit_widths: torch.Tensor) -> bytes:
    """Numpy-accelerated bit packing."""
    import numpy as np
    
    idx_np = indices.cpu().numpy().astype(np.int64)
    bw_list = bit_widths.tolist()
    num_rows, num_components = idx_np.shape
    
    accumulator = 0
    bits_in_acc = 0
    output = bytearray()
    
    for comp_idx in range(num_components):
        width = int(bw_list[comp_idx])
        if width == 0:
            continue
        mask = (1 << width) - 1
        col = idx_np[:, comp_idx]
        for val in col:
            accumulator |= (int(val) & mask) << bits_in_acc
            bits_in_acc += width
            while bits_in_acc >= 8:
                output.append(accumulator & 0xFF)
                accumulator >>= 8
                bits_in_acc -= 8
    
    if bits_in_acc:
        output.append(accumulator & 0xFF)
    
    return bytes(output)


def _pack_bits_python(indices: torch.Tensor, bit_widths: torch.Tensor) -> bytes:
    """Pure Python fallback."""
    bw_list = bit_widths.tolist()
    num_rows, num_components = indices.shape
    
    accumulator = 0
    bits_in_acc = 0
    output = bytearray()
    
    for comp_idx in range(num_components):
        width = int(bw_list[comp_idx])
        if width == 0:
            continue
        mask = (1 << width) - 1
        col = indices[:, comp_idx].tolist()
        for val in col:
            accumulator |= (int(val) & mask) << bits_in_acc
            bits_in_acc += width
            while bits_in_acc >= 8:
                output.append(accumulator & 0xFF)
                accumulator >>= 8
                bits_in_acc -= 8
    
    if bits_in_acc:
        output.append(accumulator & 0xFF)
    
    return bytes(output)


def fast_unpack_dequantize(
    packed_bytes: bytes,
    bit_widths: torch.Tensor,
    num_rows: int,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Unpack and dequantize — uses GPU-accelerated version with vectorized dequant per column."""
    try:
        from .triton_kernels import gpu_unpack_dequantize
        return gpu_unpack_dequantize(packed_bytes, bit_widths, num_rows, scales, zero_points)
    except ImportError:
        pass
    
    # Fallback: pure Python
    num_components = bit_widths.numel()
    bw_list = bit_widths.tolist()
    
    data = list(packed_bytes)
    byte_idx = 0
    accumulator = 0
    bits_in_acc = 0
    
    result = torch.zeros(num_rows, num_components, dtype=torch.float32)
    
    for comp_idx in range(num_components):
        width = int(bw_list[comp_idx])
        if width == 0:
            continue
        mask = (1 << width) - 1
        s = float(scales[comp_idx].item())
        zp = float(zero_points[comp_idx].item())
        
        for row in range(num_rows):
            while bits_in_acc < width:
                if byte_idx >= len(data):
                    raise ValueError("Insufficient packed data.")
                accumulator |= data[byte_idx] << bits_in_acc
                bits_in_acc += 8
                byte_idx += 1
            val = accumulator & mask
            accumulator >>= width
            bits_in_acc -= width
            result[row, comp_idx] = (val - zp) * s
    
    return result
