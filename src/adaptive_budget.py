"""Per-layer, per-kind adaptive bit budget allocation for KVTC.

Instead of uniform bits across all layers, allocate more bits to
high-entropy layers and fewer to low-entropy ones. The total bit
budget is preserved — bits are redistributed, not added.

Key insight: different layers have wildly different compressibility.
Layer 27 keys might compress at 0.5 bits while layer 25 values need 5 bits.
Giving them the same budget wastes bits on easy layers and starves hard ones.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple

from common import CalibrationData, PCAEntry


def compute_layer_difficulty(calibration: CalibrationData) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-layer compression difficulty from calibration eigenvalues.
    
    Difficulty is based on the "flatness" of the eigenvalue spectrum.
    A flat spectrum (all eigenvalues similar) = high entropy = hard to compress.
    A steep spectrum (few dominant eigenvalues) = low entropy = easy to compress.
    
    Returns:
        (key_difficulty, val_difficulty) arrays of shape [num_layers]
        Normalized so mean = 1.0. Values > 1 are harder than average.
    """
    # Collect eigenvalue spectra per layer
    key_entropies = []
    val_entropies = []
    
    # Find number of layers
    max_layer = max(k[0] for k in calibration.entries.keys()) + 1
    
    for layer_idx in range(max_layer):
        for kind, collector in [("keys", key_entropies), ("values", val_entropies)]:
            # Find entries for this layer+kind
            entries = [(k, v) for k, v in calibration.entries.items() 
                      if k[0] == layer_idx and k[2] == kind]
            
            if not entries:
                collector.append(1.0)
                continue
            
            # Average eigenvalue entropy across head groups
            total_entropy = 0
            for _, entry in entries:
                ev = entry.eigenvalues.float()
                # Normalized eigenvalues as probability distribution
                ev_norm = ev / ev.sum().clamp(min=1e-10)
                # Shannon entropy: higher = more uniform = harder to compress
                entropy = -(ev_norm * (ev_norm + 1e-10).log()).sum().item()
                total_entropy += entropy
            
            collector.append(total_entropy / len(entries))
    
    key_diff = np.array(key_entropies)
    val_diff = np.array(val_entropies)
    
    # Normalize to mean = 1.0
    key_diff = key_diff / max(key_diff.mean(), 1e-10)
    val_diff = val_diff / max(val_diff.mean(), 1e-10)
    
    return key_diff, val_diff


def compute_optimal_per_layer_budgets(
    calibration: CalibrationData,
    key_bits_avg: float,
    value_bits_avg: float,
    key_difficulty: np.ndarray,
    val_difficulty: np.ndarray,
    min_bits: float = 0.5,
    max_bits: float = 12.0,
) -> Dict[Tuple[int, int, str], int]:
    """Compute optimal per-layer bit budgets that preserve the total budget.
    
    The total bits across all layers equals what uniform allocation would give,
    but individual layers can vary based on their difficulty.
    
    Args:
        calibration: CalibrationData with entries
        key_bits_avg: Average bits per component for keys
        value_bits_avg: Average bits per component for values
        key_difficulty: Per-layer key difficulty (mean=1.0)
        val_difficulty: Per-layer value difficulty (mean=1.0)
        min_bits: Minimum bits per component (floor)
        max_bits: Maximum bits per component (ceiling)
    
    Returns:
        Dict mapping (layer_idx, group_idx, kind) -> bit_budget (int)
    """
    dim = None
    for entry in calibration.entries.values():
        dim = entry.eigenvectors.shape[0]
        break
    
    if dim is None:
        return {}
    
    num_layers = len(key_difficulty)
    budgets = {}
    
    # For each kind, redistribute bits proportional to difficulty
    for kind, base_bits, difficulty in [
        ("keys", key_bits_avg, key_difficulty),
        ("values", value_bits_avg, val_difficulty),
    ]:
        # Target total budget
        total_budget = base_bits * num_layers * dim
        
        # Allocate proportional to difficulty (harder layers get more)
        # Use sqrt of difficulty to dampen extreme allocations
        weights = np.sqrt(difficulty)
        weights = weights / weights.sum() * num_layers  # Normalize to sum=num_layers
        
        per_layer_bits = weights * base_bits
        
        # Clamp to [min_bits, max_bits]
        per_layer_bits = np.clip(per_layer_bits, min_bits, max_bits)
        
        # Adjust to match total budget exactly
        current_total = per_layer_bits.sum() * dim
        scale = total_budget / max(current_total, 1)
        per_layer_bits = per_layer_bits * scale
        per_layer_bits = np.clip(per_layer_bits, min_bits, max_bits)
        
        # Assign to entries
        for layer_idx in range(num_layers):
            for key, entry in calibration.entries.items():
                if key[0] == layer_idx and key[2] == kind:
                    budgets[key] = max(1, int(round(per_layer_bits[layer_idx] * dim)))
    
    return budgets


def apply_adaptive_budgets(
    calibration: CalibrationData,
    key_bits_avg: float,
    value_bits_avg: float,
    strength: float = 1.0,
):
    """Apply per-layer adaptive budgets to calibration entries in-place.
    
    Args:
        calibration: CalibrationData to modify
        key_bits_avg: Average bits for keys
        value_bits_avg: Average bits for values  
        strength: 0.0 = uniform (no adaptation), 1.0 = fully adaptive
    """
    key_diff, val_diff = compute_layer_difficulty(calibration)
    
    if strength <= 0:
        # Just set uniform
        dim = None
        for entry in calibration.entries.values():
            dim = entry.eigenvectors.shape[0]
            break
        for (li, gi, kind), entry in calibration.entries.items():
            entry.bit_budget = int(dim * (key_bits_avg if kind == "keys" else value_bits_avg))
        return key_diff, val_diff
    
    optimal = compute_optimal_per_layer_budgets(
        calibration, key_bits_avg, value_bits_avg, key_diff, val_diff
    )
    
    dim = None
    for entry in calibration.entries.values():
        dim = entry.eigenvectors.shape[0]
        break
    
    # Blend between uniform and adaptive based on strength
    for key, entry in calibration.entries.items():
        kind = key[2]
        uniform_budget = int(dim * (key_bits_avg if kind == "keys" else value_bits_avg))
        adaptive_budget = optimal.get(key, uniform_budget)
        
        blended = int(round(uniform_budget * (1 - strength) + adaptive_budget * strength))
        entry.bit_budget = max(1, blended)
    
    return key_diff, val_diff


def print_budget_summary(calibration: CalibrationData):
    """Print per-layer budget allocation for debugging."""
    max_layer = max(k[0] for k in calibration.entries.keys()) + 1
    
    print(f"\n  Per-Layer Bit Budgets:")
    print(f"  {'Layer':>5s} | {'K budget':>8s} | {'K bits/d':>7s} | {'V budget':>8s} | {'V bits/d':>7s}")
    print(f"  {'-'*5:s}-+-{'-'*8:s}-+-{'-'*7:s}-+-{'-'*8:s}-+-{'-'*7:s}")
    
    dim = None
    for entry in calibration.entries.values():
        dim = entry.eigenvectors.shape[0]
        break
    
    for li in range(max_layer):
        k_budget = None
        v_budget = None
        for key, entry in calibration.entries.items():
            if key[0] == li:
                if key[2] == "keys":
                    k_budget = entry.bit_budget
                else:
                    v_budget = entry.bit_budget
        
        k_bits = f"{k_budget/dim:.1f}" if k_budget and dim else "?"
        v_bits = f"{v_budget/dim:.1f}" if v_budget and dim else "?"
        print(f"  {li:5d} | {k_budget or 0:8d} | {k_bits:>7s} | {v_budget or 0:8d} | {v_bits:>7s}")
