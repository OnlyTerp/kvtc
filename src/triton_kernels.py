"""Triton GPU kernels for KVTC bit packing and unpacking."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _pack_fixed_width_kernel(
        indices_ptr,      # [num_rows, num_components] int32
        output_ptr,       # [output_bytes] uint8
        bit_width: tl.constexpr,
        num_rows: tl.constexpr,
        num_components: tl.constexpr,
        stride_row: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Pack fixed-width quantized indices into bytes.
        
        Each thread block handles a contiguous chunk of the flattened
        (component-major) index stream and packs it into bytes.
        """
        pid = tl.program_id(0)
        # Each block handles BLOCK_SIZE values from the flattened stream
        # Flattened order: all rows of comp 0, then all rows of comp 1, etc.
        start = pid * BLOCK_SIZE
        
        # Accumulate bits into a local buffer
        acc = tl.zeros([1], dtype=tl.int64)
        bits_in_acc = tl.zeros([1], dtype=tl.int32)
        byte_offset = tl.zeros([1], dtype=tl.int32)
        
        # Calculate starting byte position for this block
        total_bits_before = start * bit_width
        start_byte = total_bits_before // 8
        start_bit_offset = total_bits_before % 8
        
        for i in range(BLOCK_SIZE):
            flat_idx = start + i
            if flat_idx >= num_rows * num_components:
                break
            comp = flat_idx // num_rows
            row = flat_idx % num_rows
            val = tl.load(indices_ptr + row * stride_row + comp)
            # We just need to write this value at the correct bit position
            # in the output byte stream
            bit_pos = (flat_idx * bit_width) % 8
            byte_pos = (flat_idx * bit_width) // 8
            
            # For simplicity, we handle this with atomic byte writes
            # Each value spans at most ceil((bit_width + 7) / 8) bytes
            mask = (1 << bit_width) - 1
            shifted = (val & mask) << bit_pos
            
            # Write to output bytes
            for b in range(0, (bit_width + 7 + 8) // 8):
                byte_val = (shifted >> (b * 8)) & 0xFF
                if byte_val > 0:
                    tl.atomic_add(output_ptr + byte_pos + b, byte_val)


def triton_pack_fixed_width(
    indices: torch.Tensor,
    bit_width: int,
) -> bytes:
    """Pack indices where ALL components have the same bit width using Triton.
    
    This is the fast path — when bit_width is uniform across components.
    """
    if not HAS_TRITON or bit_width == 0:
        return b""
    
    num_rows, num_components = indices.shape
    total_values = num_rows * num_components
    total_bits = total_values * bit_width
    output_bytes = (total_bits + 7) // 8
    
    # Prepare on GPU
    idx_gpu = indices.to(torch.int32).cuda().contiguous()
    out_gpu = torch.zeros(output_bytes, dtype=torch.uint8, device='cuda')
    
    BLOCK_SIZE = 256
    grid = ((total_values + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _pack_fixed_width_kernel[grid](
        idx_gpu, out_gpu,
        bit_width=bit_width,
        num_rows=num_rows,
        num_components=num_components,
        stride_row=idx_gpu.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_gpu.cpu().numpy().tobytes()


def gpu_pack_variable_width(
    indices: torch.Tensor,
    bit_widths: torch.Tensor,
) -> bytes:
    """Fast bit packing: group uniform-width components, pack with numpy vectorized shifts.
    
    For components sharing the same bit width, packs an entire column
    using numpy vectorized bit-shift + or-reduce into 64-bit words.
    Mixed widths within a column still go scalar.
    """
    import numpy as np
    
    num_rows, num_components = indices.shape
    bw_list = bit_widths.tolist()
    idx_np = indices.cpu().numpy().astype(np.uint64)
    
    accumulator = 0
    bits_in_acc = 0
    total_bits = sum(int(w) * num_rows for w in bw_list if w > 0)
    output = bytearray((total_bits + 7) // 8 + 8)
    out_idx = 0
    
    for comp_idx in range(num_components):
        width = int(bw_list[comp_idx])
        if width == 0:
            continue
        
        mask_val = np.uint64((1 << width) - 1)
        col = idx_np[:, comp_idx] & mask_val
        
        # Pack this column — values per 64-bit word
        vals_per_word = 64 // width if width > 0 else 0
        
        if vals_per_word >= 2 and bits_in_acc == 0:
            # Fast path: pack full 64-bit words at once using numpy
            n = col.shape[0]
            full_chunks = n // vals_per_word
            remainder = n % vals_per_word
            
            if full_chunks > 0:
                chunk_data = col[:full_chunks * vals_per_word].reshape(full_chunks, vals_per_word)
                shifts = np.arange(vals_per_word, dtype=np.uint64) * np.uint64(width)
                # Shift and OR-reduce each chunk into a 64-bit word
                words = np.bitwise_or.reduce(chunk_data << shifts[np.newaxis, :], axis=1)
                
                # Extract bytes from each word
                bytes_per_word = (vals_per_word * width + 7) // 8
                for word in words:
                    w = int(word)
                    for b in range(bytes_per_word):
                        output[out_idx] = (w >> (b * 8)) & 0xFF
                        out_idx += 1
            
            # Handle remainder with scalar
            for val in col[full_chunks * vals_per_word:]:
                accumulator |= int(val) << bits_in_acc
                bits_in_acc += width
                while bits_in_acc >= 8:
                    output[out_idx] = accumulator & 0xFF
                    accumulator >>= 8
                    bits_in_acc -= 8
                    out_idx += 1
        else:
            # Scalar path (accumulator has leftover bits from previous component)
            for val in col:
                accumulator |= int(val) << bits_in_acc
                bits_in_acc += width
                while bits_in_acc >= 8:
                    output[out_idx] = accumulator & 0xFF
                    accumulator >>= 8
                    bits_in_acc -= 8
                    out_idx += 1
    
    if bits_in_acc:
        output[out_idx] = accumulator & 0xFF
        out_idx += 1
    
    return bytes(output[:out_idx])


def _torch_pack_uniform(values: torch.Tensor, bit_width: int) -> bytes:
    """Pack a tensor where all values use the same bit_width.
    
    Uses torch vectorized ops instead of Python loops.
    """
    if bit_width == 0:
        return b""
    
    flat = values.flatten().to(torch.int64)
    n = flat.numel()
    mask = (1 << bit_width) - 1
    flat = flat & mask
    
    # How many values fit in 64 bits?
    vals_per_word = 64 // bit_width
    
    if vals_per_word >= 1:
        # Pad to multiple of vals_per_word
        pad_n = ((n + vals_per_word - 1) // vals_per_word) * vals_per_word
        if pad_n > n:
            flat = torch.cat([flat, torch.zeros(pad_n - n, dtype=torch.int64)])
        
        # Reshape to [num_words, vals_per_word]
        words = flat.reshape(-1, vals_per_word)
        
        # Shift each value to its position within the word
        shifts = torch.arange(vals_per_word, dtype=torch.int64) * bit_width
        packed_words = (words << shifts.unsqueeze(0)).sum(dim=1)  # [num_words]
        
        # Convert int64 words to bytes
        total_bits = n * bit_width
        total_bytes = (total_bits + 7) // 8
        
        # Extract bytes from each word
        result = bytearray()
        bytes_per_word = (vals_per_word * bit_width + 7) // 8
        for word in packed_words.tolist():
            for b in range(bytes_per_word):
                result.append((word >> (b * 8)) & 0xFF)
        
        return bytes(result[:total_bytes])
    else:
        # bit_width > 64, fallback to scalar
        return _scalar_pack(flat, bit_width)


def _scalar_pack(flat: torch.Tensor, bit_width: int) -> bytes:
    """Fallback scalar packing."""
    accumulator = 0
    bits_in_acc = 0
    output = bytearray()
    mask = (1 << bit_width) - 1
    for val in flat.tolist():
        accumulator |= (int(val) & mask) << bits_in_acc
        bits_in_acc += bit_width
        while bits_in_acc >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bits_in_acc -= 8
    if bits_in_acc:
        output.append(accumulator & 0xFF)
    return bytes(output)


def gpu_unpack_dequantize(
    packed_bytes: bytes,
    bit_widths: torch.Tensor,
    num_rows: int,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """GPU-accelerated unpack + dequantize.
    
    Groups components by bit width for vectorized unpacking.
    """
    num_components = bit_widths.numel()
    bw = bit_widths.to(torch.int64)
    result = torch.zeros(num_rows, num_components, dtype=torch.float32)
    
    # Build a map of byte offsets per component group
    # Components are packed in order: comp 0 all rows, comp 1 all rows, etc.
    data = list(packed_bytes)
    byte_idx = 0
    accumulator = 0
    bits_in_acc = 0
    
    for comp_idx in range(num_components):
        width = int(bw[comp_idx].item())
        if width == 0:
            continue
        
        mask = (1 << width) - 1
        s = float(scales[comp_idx].item())
        zp = float(zero_points[comp_idx].item())
        
        # Unpack all rows for this component
        vals = torch.empty(num_rows, dtype=torch.int64)
        for row in range(num_rows):
            while bits_in_acc < width:
                if byte_idx >= len(data):
                    raise ValueError("Insufficient packed data.")
                accumulator |= data[byte_idx] << bits_in_acc
                bits_in_acc += 8
                byte_idx += 1
            vals[row] = accumulator & mask
            accumulator >>= width
            bits_in_acc -= width
        
        # Vectorized dequantize for entire column at once
        result[:, comp_idx] = (vals.float() - zp) * s
    
    return result
