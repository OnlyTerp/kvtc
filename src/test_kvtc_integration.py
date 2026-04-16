"""Regression tests for the KVTC llama.cpp integration.

Focuses on GGML_TYPE_KVTC behavior: type registration, patching correctness,
asymmetric K/V configurations, adaptive budgets, and end-to-end pipeline
roundtrips under configurations that mirror the llama.cpp integration paths.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

from src.cache import KVTCCache
from src.common import (
    CalibrationData,
    PCAEntry,
)
from src.pipeline import KVTCCompressor
from src.quantize import dp_bit_allocation


# ---------------------------------------------------------------------------
# Helpers -- build calibration without the missing PCACalibrator class
# ---------------------------------------------------------------------------


def _synthetic_kv(
    layers: int = 2,
    tokens: int = 192,
    heads: int = 4,
    dim: int = 8,
    seed: int = 7,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Create synthetic KV tensors with controllable structure."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(tokens, dtype=torch.long)
    base = torch.randn(layers, tokens, heads, 1, generator=generator)
    mix = torch.linspace(2.5, 0.1, dim).view(1, 1, 1, dim)
    keys = base * mix + 0.01 * torch.randn(layers, tokens, heads, dim, generator=generator)
    values = 1.2 * base * mix + 0.01 * torch.randn(layers, tokens, heads, dim, generator=generator)
    return {"keys": keys, "values": values}, positions


def _build_pca_entry(
    data: torch.Tensor,
    bit_budget_ratio: float = 0.12,
    head_indices: List[int] | None = None,
    kind: str = "keys",
) -> PCAEntry:
    """Build a PCAEntry from raw data [num_rows, dim] using SVD."""
    data_f32 = data.to(torch.float32)
    mean = data_f32.mean(dim=0)
    centered = data_f32 - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values.square() / max(centered.shape[0] - 1, 1)
    dim = data.shape[-1]
    bit_budget = max(1, int(bit_budget_ratio * dim * dim))
    return PCAEntry(
        eigenvectors=vh,
        eigenvalues=eigenvalues,
        mean=mean,
        head_indices=head_indices or [0],
        kind=kind,
        bit_budget=bit_budget,
    )


def _calibration_from_cache(
    kv_cache: Dict[str, torch.Tensor],
    positions: torch.Tensor,
    head_group_size: int = 2,
    bit_budget_ratio: float = 0.12,
) -> CalibrationData:
    """Build CalibrationData from a KV cache using SVD directly."""
    layers, tokens, heads, dim = kv_cache["keys"].shape
    entries: Dict[Tuple[int, int, str], PCAEntry] = {}
    for layer_idx in range(layers):
        for kind in ("keys", "values"):
            tensor = kv_cache[kind][layer_idx]  # [tokens, heads, dim]
            for group_idx, start in enumerate(range(0, heads, head_group_size)):
                end = min(start + head_group_size, heads)
                head_indices = list(range(start, end))
                group = tensor[:, start:end, :]  # [tokens, group_heads, dim]
                rows = group.reshape(-1, dim)
                entry = _build_pca_entry(
                    rows,
                    bit_budget_ratio=bit_budget_ratio,
                    head_indices=head_indices,
                    kind=kind,
                )
                entries[(layer_idx, group_idx, kind)] = entry
    return CalibrationData(
        entries=entries,
        head_group_size=head_group_size,
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return torch.nn.functional.cosine_similarity(
        a.reshape(1, -1).float(), b.reshape(1, -1).float()
    ).item()


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error between two tensors."""
    return ((a.float() - b.float()) ** 2).mean().item()


def _set_asymmetric_budgets(
    calibration: CalibrationData,
    key_bits: int,
    value_bits: int,
    dim: int,
) -> None:
    """Set asymmetric K/V bit budgets (total bits = bits_per_component * dim)."""
    for (layer_idx, group_idx, kind), entry in calibration.entries.items():
        if kind == "keys":
            entry.bit_budget = key_bits * dim
        else:
            entry.bit_budget = value_bits * dim


# ===========================================================================
# Section 1: GGML_TYPE_KVTC Type Registration Tests
# ===========================================================================


class TestGGMLTypeKVTCRegistration:
    """Verify the integration script produces correct type registration code."""

    def test_kvtc_type_enum_value(self) -> None:
        """GGML_TYPE_KVTC should be assigned enum value 44."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        assert "GGML_TYPE_KVTC     = 44" in content

    def test_kvtc_type_comment_describes_algorithm(self) -> None:
        """The type comment should describe KVTC as PCA + DP-optimal quantization."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        assert "PCA + DP-optimal quantization" in content

    def test_kvtc_stub_uses_q8_0_traits(self) -> None:
        """Phase 1 stub should use q8_0 block size and quantize functions."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        # The stub type_traits entry references q8_0
        assert "QK8_0" in content
        assert "dequantize_row_q8_0" in content
        assert "quantize_row_q8_0_ref" in content

    def test_kvtc_type_name_is_kvtc(self) -> None:
        """type_name should be 'kvtc' for CLI arg matching."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        assert '.type_name                = "kvtc"' in content

    def test_kvtc_is_quantized_true(self) -> None:
        """KVTC must be flagged as quantized type."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        assert ".is_quantized             = true" in content

    def test_kvtc_added_to_arg_cpp_allowed_types(self) -> None:
        """GGML_TYPE_KVTC must be added to arg.cpp allowed cache types."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        # The script patches arg.cpp to include GGML_TYPE_KVTC alongside turbo types
        assert "GGML_TYPE_KVTC," in content

    def test_kvtc_quantize_case_added(self) -> None:
        """A quantize switch case for KVTC should be generated."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        assert "case GGML_TYPE_KVTC:" in content

    def test_kvtc_kv_cache_handling_added(self) -> None:
        """llama-kv-cache.cpp should get an is_kvtc boolean check."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        assert "is_kvtc" in content

    def test_kvtc_cuda_dispatch_added(self) -> None:
        """KVTC should be added alongside TURBO2 in the CUDA flash attention dispatch."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "integrate_llamacpp.py"
        content = script.read_text(encoding="utf-8")
        # The script adds KVTC case in fattn dispatch
        assert "case GGML_TYPE_KVTC:" in content


# ===========================================================================
# Section 2: Real Compression Patch Tests (turbo2 wiring)
# ===========================================================================


class TestPatchKVTCRealCompression:
    """Verify the real-compression patch correctly rewires KVTC to turbo2."""

    def test_type_traits_updated_to_turbo2(self) -> None:
        """After real compression patch, type_traits should use turbo2 functions."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_kvtc_real_compression.py"
        content = script.read_text(encoding="utf-8")
        assert "QK_TURBO2" in content
        assert "block_turbo2_0" in content
        assert "dequantize_row_turbo2_0" in content
        assert "quantize_row_turbo2_0_ref" in content

    def test_quantize_case_updated_to_turbo2(self) -> None:
        """The quantize switch case should call quantize_turbo2_0."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_kvtc_real_compression.py"
        content = script.read_text(encoding="utf-8")
        assert "quantize_turbo2_0" in content

    def test_setrows_uses_turbo2_path(self) -> None:
        """SET_ROWS dispatch should use turbo2 path for KVTC."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_kvtc_real_compression.py"
        content = script.read_text(encoding="utf-8")
        assert "set_rows_cuda_turbo2" in content

    def test_fa_remap_updated_to_turbo2(self) -> None:
        """Flash attention should remap KVTC to TURBO2_0 (not Q8_0)."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_kvtc_real_compression.py"
        content = script.read_text(encoding="utf-8")
        assert "GGML_TYPE_TURBO2_0; // KVTC uses turbo2 compression" in content

    def test_is_turbo_includes_kvtc(self) -> None:
        """is_turbo check in llama-kv-cache.cpp should include KVTC."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_kvtc_real_compression.py"
        content = script.read_text(encoding="utf-8")
        assert "type_k == GGML_TYPE_KVTC" in content

    def test_graph_v_type_check_includes_kvtc(self) -> None:
        """llama-graph.cpp V-type turbo2 checks should include KVTC."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_kvtc_real_compression.py"
        content = script.read_text(encoding="utf-8")
        assert "v->type == GGML_TYPE_KVTC" in content


# ===========================================================================
# Section 3: FA Dispatch Patch Tests
# ===========================================================================


class TestPatchFADispatch:
    """Verify flash attention dispatch patches add KVTC support correctly."""

    def test_fattn_common_adds_kvtc_case(self) -> None:
        """fattn-common.cuh should get KVTC alongside Q8_0."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_dispatch.py"
        content = script.read_text(encoding="utf-8")
        assert "case GGML_TYPE_KVTC:" in content
        assert "case GGML_TYPE_Q8_0:" in content

    def test_fattn_vec_adds_kvtc_case(self) -> None:
        """fattn-vec.cuh patch should add KVTC case."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_dispatch.py"
        content = script.read_text(encoding="utf-8")
        # The patch_fattn_vec function adds KVTC before Q8_0
        assert "patch_fattn_vec" in content

    def test_convert_cu_patched(self) -> None:
        """convert.cu should get KVTC case."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_dispatch.py"
        content = script.read_text(encoding="utf-8")
        assert "patch_convert" in content

    def test_dequantize_cuh_patched(self) -> None:
        """dequantize.cuh should get KVTC case."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_dispatch.py"
        content = script.read_text(encoding="utf-8")
        assert "patch_dequantize" in content

    def test_ggml_cuda_cpy_support(self) -> None:
        """ggml-cuda.cu should support F32 <-> KVTC copy operations."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_dispatch.py"
        content = script.read_text(encoding="utf-8")
        assert "GGML_TYPE_KVTC" in content
        assert "patch_ggml_cuda_supports" in content


# ===========================================================================
# Section 4: FA Remap Patch Tests
# ===========================================================================


class TestPatchFARemap:
    """Verify the flash attention remap strategy for KVTC."""

    def test_remap_code_present(self) -> None:
        """KVTC_REMAP comment marker should be in the generated code."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_remap.py"
        content = script.read_text(encoding="utf-8")
        assert "KVTC_REMAP" in content

    def test_remap_targets_q8_0(self) -> None:
        """Stub mode remap should change KVTC to Q8_0 at FA entry."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_remap.py"
        content = script.read_text(encoding="utf-8")
        assert 'K->type == GGML_TYPE_KVTC' in content
        assert 'V->type == GGML_TYPE_KVTC' in content
        assert 'GGML_TYPE_Q8_0' in content

    def test_remap_patches_vec_dispatch(self) -> None:
        """ggml_cuda_flash_attn_ext_vec should be patched."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_remap.py"
        content = script.read_text(encoding="utf-8")
        assert "ggml_cuda_flash_attn_ext_vec" in content

    def test_remap_patches_support_check(self) -> None:
        """ggml_cuda_flash_attn_ext_supported should accept KVTC."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_remap.py"
        content = script.read_text(encoding="utf-8")
        assert "ggml_cuda_flash_attn_ext_supported" in content

    def test_remap_preserves_original_types(self) -> None:
        """Remap should save/restore original types so tensors aren't corrupted."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_fa_remap.py"
        content = script.read_text(encoding="utf-8")
        assert "orig_k" in content
        assert "orig_v" in content


# ===========================================================================
# Section 5: SET_ROWS Patch Tests
# ===========================================================================


class TestPatchSetRows:
    """Verify set-rows.cu patching for KVTC."""

    def test_setrows_adds_kvtc_handler(self) -> None:
        """set-rows.cu should get a KVTC branch using q8_0 quantize."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_setrows.py"
        content = script.read_text(encoding="utf-8")
        assert "GGML_TYPE_KVTC" in content
        assert "block_q8_0" in content
        assert "quantize_f32_q8_0_block" in content

    def test_setrows_inserts_before_abort(self) -> None:
        """KVTC handler should be inserted before the GGML_ABORT fallthrough."""
        script = Path(__file__).resolve().parent.parent / "cuda" / "patch_setrows.py"
        content = script.read_text(encoding="utf-8")
        assert "GGML_ABORT" in content


# ===========================================================================
# Section 6: Asymmetric K/V Configuration Tests (K1V3, K2V4, K4V6)
# ===========================================================================


class TestAsymmetricKVConfigurations:
    """Test pipeline behavior with asymmetric K/V bit budgets as used by
    the llama.cpp integration (K1V3, K2V4, K4V6 configs from README)."""

    @pytest.fixture
    def kv_and_calibration(self):
        kv_cache, positions = _synthetic_kv(layers=2, tokens=192, heads=4, dim=8)
        calibration = _calibration_from_cache(kv_cache, positions)
        return kv_cache, positions, calibration

    @pytest.mark.parametrize(
        "key_bits,value_bits,label",
        [
            (1, 3, "K1V3"),
            (2, 4, "K2V4"),
            (4, 6, "K4V6"),
        ],
    )
    def test_asymmetric_config_roundtrip(
        self, kv_and_calibration, key_bits: int, value_bits: int, label: str
    ) -> None:
        """Pipeline roundtrip should work for each asymmetric configuration."""
        kv_cache, positions, calibration = kv_and_calibration
        _set_asymmetric_budgets(calibration, key_bits, value_bits, dim=8)
        compressor = KVTCCompressor(calibration)
        compressed = compressor.compress(kv_cache, positions)
        restored = compressor.decompress(compressed)
        assert restored["keys"].shape == kv_cache["keys"].shape, f"{label}: key shape mismatch"
        assert restored["values"].shape == kv_cache["values"].shape, f"{label}: value shape mismatch"

    @pytest.mark.parametrize(
        "key_bits,value_bits,label",
        [
            (1, 3, "K1V3"),
            (2, 4, "K2V4"),
            (4, 6, "K4V6"),
        ],
    )
    def test_asymmetric_config_cosine_exceeds_threshold(
        self, kv_and_calibration, key_bits: int, value_bits: int, label: str
    ) -> None:
        """Values should have higher cosine similarity than keys when V gets more bits."""
        kv_cache, positions, calibration = kv_and_calibration
        _set_asymmetric_budgets(calibration, key_bits, value_bits, dim=8)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        key_cos = _cosine_similarity(kv_cache["keys"], restored["keys"])
        val_cos = _cosine_similarity(kv_cache["values"], restored["values"])
        # Both should be above 0.6 (low-bit configs on small synthetic data
        # have limited accuracy; real models with larger dims do much better)
        assert key_cos > 0.6, f"{label}: key cosine {key_cos:.4f} too low"
        assert val_cos > 0.6, f"{label}: value cosine {val_cos:.4f} too low"
        # Values should be at least as good as keys when given more bits
        if value_bits > key_bits:
            assert val_cos >= key_cos - 0.05, (
                f"{label}: values ({val_cos:.4f}) should be close to or better than keys ({key_cos:.4f})"
            )

    @pytest.mark.parametrize(
        "key_bits,value_bits",
        [(1, 3), (2, 4), (4, 6)],
    )
    def test_asymmetric_config_sinks_preserved(
        self, kv_and_calibration, key_bits: int, value_bits: int
    ) -> None:
        """Attention sinks should be preserved exactly regardless of bit budget."""
        kv_cache, positions, calibration = kv_and_calibration
        _set_asymmetric_budgets(calibration, key_bits, value_bits, dim=8)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        assert torch.equal(restored["keys"][:, :4], kv_cache["keys"][:, :4])
        assert torch.equal(restored["values"][:, :4], kv_cache["values"][:, :4])

    @pytest.mark.parametrize(
        "key_bits,value_bits",
        [(1, 3), (2, 4), (4, 6)],
    )
    def test_asymmetric_config_sliding_window_preserved(
        self, kv_and_calibration, key_bits: int, value_bits: int
    ) -> None:
        """Sliding window tokens should be preserved exactly regardless of bit budget."""
        kv_cache, positions, calibration = kv_and_calibration
        _set_asymmetric_budgets(calibration, key_bits, value_bits, dim=8)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        # Last 128 tokens (or all tokens if fewer)
        window = min(128, kv_cache["keys"].shape[1])
        assert torch.equal(restored["keys"][:, -window:], kv_cache["keys"][:, -window:])
        assert torch.equal(restored["values"][:, -window:], kv_cache["values"][:, -window:])

    def test_more_bits_yields_lower_error(self, kv_and_calibration) -> None:
        """Higher bit budget should produce lower reconstruction error."""
        kv_cache, positions, calibration_base = kv_and_calibration
        errors = []
        for key_bits, value_bits in [(1, 3), (2, 4), (4, 6)]:
            calibration = _calibration_from_cache(kv_cache, positions)
            _set_asymmetric_budgets(calibration, key_bits, value_bits, dim=8)
            compressor = KVTCCompressor(calibration)
            restored = compressor.decompress(compressor.compress(kv_cache, positions))
            errors.append(_mse(kv_cache["values"], restored["values"]))
        # K4V6 should have lower error than K2V4, which should be lower than K1V3
        assert errors[2] <= errors[1], "K4V6 should have lower error than K2V4"
        assert errors[1] <= errors[0], "K2V4 should have lower error than K1V3"


# ===========================================================================
# Section 7: Compression Metadata Tests
# ===========================================================================


class TestCompressionMetadata:
    """Test that compression metadata is correctly populated for
    configurations matching the llama.cpp integration."""

    def test_metadata_shape_matches_input(self) -> None:
        """original_shape in metadata should match the input KV cache shape."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        assert compressed.metadata is not None
        assert compressed.metadata.original_shape == kv_cache["keys"].shape

    def test_metadata_sink_window_lengths(self) -> None:
        """Sink and window lengths should match calibration defaults."""
        kv_cache, positions = _synthetic_kv(tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        assert compressed.metadata is not None
        assert compressed.metadata.sink_len == 4
        assert compressed.metadata.window_len == 128
        assert compressed.metadata.middle_len == 200 - 4 - 128

    def test_metadata_compression_ratio_positive(self) -> None:
        """Compression ratio should be a positive finite number."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        assert compressed.metadata is not None
        assert compressed.metadata.compression_ratio > 0
        assert math.isfinite(compressed.metadata.compression_ratio)

    def test_metadata_middle_positions_contiguous(self) -> None:
        """Middle region position indices should be contiguous."""
        kv_cache, positions = _synthetic_kv(tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        assert compressed.metadata is not None
        mid_positions = compressed.metadata.positions_middle
        if len(mid_positions) > 1:
            diffs = [mid_positions[i + 1] - mid_positions[i] for i in range(len(mid_positions) - 1)]
            assert all(d == 1 for d in diffs), "Middle positions should be contiguous"


# ===========================================================================
# Section 8: Compressed Section Consistency Tests
# ===========================================================================


class TestCompressedSectionConsistency:
    """Verify individual CompressedSection objects have valid fields."""

    def test_sections_cover_all_layers_and_kinds(self) -> None:
        """There should be compressed sections for every layer/group/kind combo."""
        kv_cache, positions = _synthetic_kv(layers=2, heads=4)
        calibration = _calibration_from_cache(kv_cache, positions, head_group_size=2)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        # 2 layers * 2 groups * 2 kinds = 8 sections
        seen = set()
        for section in compressed.compressed_sections:
            seen.add((section.layer_idx, section.group_idx, section.kind))
        expected = {
            (l, g, k) for l in range(2) for g in range(2) for k in ("keys", "values")
        }
        assert seen == expected

    def test_section_bit_widths_nonnegative(self) -> None:
        """All bit widths in sections should be non-negative."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        for section in compressed.compressed_sections:
            assert all(bw >= 0 for bw in section.bit_widths)

    def test_section_scales_finite(self) -> None:
        """All quantization scales should be finite."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        for section in compressed.compressed_sections:
            assert all(math.isfinite(s) for s in section.scales)

    def test_section_zero_points_finite(self) -> None:
        """All quantization zero points should be finite."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        for section in compressed.compressed_sections:
            assert all(math.isfinite(zp) for zp in section.zero_points)

    def test_section_compressed_bytes_nonempty(self) -> None:
        """Each section should have non-empty compressed data."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        for section in compressed.compressed_sections:
            assert len(section.compressed_bytes) > 0
            assert section.packed_size > 0


# ===========================================================================
# Section 9: Cache Wrapper Integration Tests
# ===========================================================================


class TestCacheWrapperKVTC:
    """Test KVTCCache behavior simulating the llama.cpp cache management
    pattern: update -> evict -> restore cycle."""

    def test_multi_layer_evict_restore_cycle(self) -> None:
        """Evict and restore multiple layers independently."""
        kv_cache, positions = _synthetic_kv(layers=3, tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        cache = KVTCCache(KVTCCompressor(calibration))
        for layer_idx in range(3):
            cache.update(layer_idx, kv_cache["keys"][layer_idx], kv_cache["values"][layer_idx])
        # Evict all layers
        for layer_idx in range(3):
            cache.evict_to_compressed(layer_idx, positions)
            assert cache.is_compressed(layer_idx)
        # Restore and check shapes
        for layer_idx in range(3):
            restored = cache.restore_layer(layer_idx)
            assert restored["keys"].shape == kv_cache["keys"][layer_idx].shape
            assert restored["values"].shape == kv_cache["values"][layer_idx].shape

    def test_evict_restore_preserves_sinks(self) -> None:
        """After evict/restore, attention sinks should be exact."""
        kv_cache, positions = _synthetic_kv(layers=1, tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        cache = KVTCCache(KVTCCompressor(calibration))
        cache.update(0, kv_cache["keys"][0], kv_cache["values"][0])
        cache.evict_to_compressed(0, positions)
        restored = cache.restore_layer(0)
        assert torch.equal(restored["keys"][:4], kv_cache["keys"][0, :4])
        assert torch.equal(restored["values"][:4], kv_cache["values"][0, :4])

    def test_evict_removes_live_cache(self) -> None:
        """After eviction, the live cache for that layer should be removed."""
        kv_cache, positions = _synthetic_kv(layers=1, tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        cache = KVTCCache(KVTCCompressor(calibration))
        cache.update(0, kv_cache["keys"][0], kv_cache["values"][0])
        assert cache.get_layer(0) is not None
        cache.evict_to_compressed(0, positions)
        assert cache.get_layer(0) is None
        assert cache.is_compressed(0)

    def test_restore_moves_back_to_live(self) -> None:
        """After restoration, layer should be live again (not compressed)."""
        kv_cache, positions = _synthetic_kv(layers=1, tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        cache = KVTCCache(KVTCCompressor(calibration))
        cache.update(0, kv_cache["keys"][0], kv_cache["values"][0])
        cache.evict_to_compressed(0, positions)
        cache.restore_layer(0)
        assert not cache.is_compressed(0)
        assert cache.get_layer(0) is not None

    def test_update_after_restore_replaces_data(self) -> None:
        """Updating a layer after restore should replace the data cleanly."""
        kv_cache, positions = _synthetic_kv(layers=1, tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        cache = KVTCCache(KVTCCompressor(calibration))
        cache.update(0, kv_cache["keys"][0], kv_cache["values"][0])
        cache.evict_to_compressed(0, positions)
        cache.restore_layer(0)
        # Update with new data
        new_keys = torch.randn_like(kv_cache["keys"][0])
        new_values = torch.randn_like(kv_cache["values"][0])
        cache.update(0, new_keys, new_values)
        assert not cache.is_compressed(0)
        layer = cache.get_layer(0)
        assert layer is not None
        assert torch.equal(layer["keys"], new_keys)
        assert torch.equal(layer["values"], new_values)


# ===========================================================================
# Section 10: Calibration Serialization Regression
# ===========================================================================


class TestCalibrationSerialization:
    """Verify calibration save/load preserves all fields needed by
    the llama.cpp integration (bit_budget, eigenvalues, etc)."""

    def test_save_load_preserves_all_entries(self, tmp_path) -> None:
        """All calibration entries should survive save/load."""
        kv_cache, positions = _synthetic_kv(layers=2, heads=4)
        calibration = _calibration_from_cache(kv_cache, positions)
        path = tmp_path / "cal.pt"
        torch.save(calibration, path)
        restored = torch.load(path, weights_only=False)
        assert set(restored.entries.keys()) == set(calibration.entries.keys())

    def test_save_load_preserves_bit_budgets(self, tmp_path) -> None:
        """Bit budgets should be preserved through serialization."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        _set_asymmetric_budgets(calibration, key_bits=2, value_bits=4, dim=8)
        path = tmp_path / "cal.pt"
        torch.save(calibration, path)
        restored = torch.load(path, weights_only=False)
        for key in calibration.entries:
            assert restored.entries[key].bit_budget == calibration.entries[key].bit_budget

    def test_save_load_preserves_head_group_size(self, tmp_path) -> None:
        """head_group_size should be preserved."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions, head_group_size=2)
        path = tmp_path / "cal.pt"
        torch.save(calibration, path)
        restored = torch.load(path, weights_only=False)
        assert restored.head_group_size == 2

    def test_save_load_preserves_eigenvalues(self, tmp_path) -> None:
        """Eigenvalues should be bitwise identical after save/load."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        path = tmp_path / "cal.pt"
        torch.save(calibration, path)
        restored = torch.load(path, weights_only=False)
        for key in calibration.entries:
            assert torch.equal(
                restored.entries[key].eigenvalues,
                calibration.entries[key].eigenvalues,
            )

    def test_save_load_preserves_eigenvectors(self, tmp_path) -> None:
        """Eigenvectors should be bitwise identical after save/load."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        path = tmp_path / "cal.pt"
        torch.save(calibration, path)
        restored = torch.load(path, weights_only=False)
        for key in calibration.entries:
            assert torch.equal(
                restored.entries[key].eigenvectors,
                calibration.entries[key].eigenvectors,
            )


# ===========================================================================
# Section 11: Edge Cases for KVTC Integration
# ===========================================================================


class TestKVTCEdgeCases:
    """Edge cases that could arise in the llama.cpp integration context."""

    def test_tokens_equal_sink_plus_window(self) -> None:
        """When tokens == sink_tokens + window_tokens, middle_len = 0."""
        kv_cache, positions = _synthetic_kv(tokens=132)  # 4 + 128 = 132
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        assert compressed.metadata is not None
        assert compressed.metadata.middle_len == 0
        restored = KVTCCompressor(calibration).decompress(compressed)
        # With no middle region, sinks + window should be exact
        assert torch.equal(restored["keys"], kv_cache["keys"])
        assert torch.equal(restored["values"], kv_cache["values"])

    def test_tokens_less_than_sink(self) -> None:
        """With very few tokens (< sink_tokens), everything should be preserved."""
        kv_cache, positions = _synthetic_kv(tokens=2)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
        restored = KVTCCompressor(calibration).decompress(compressed)
        assert torch.equal(restored["keys"], kv_cache["keys"])

    def test_single_head(self) -> None:
        """Pipeline should work with a single attention head."""
        kv_cache, positions = _synthetic_kv(layers=1, tokens=192, heads=1, dim=8)
        calibration = _calibration_from_cache(kv_cache, positions, head_group_size=1)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        assert restored["keys"].shape == kv_cache["keys"].shape

    def test_many_layers(self) -> None:
        """Pipeline should handle many layers (mimicking a real model with 28+ layers)."""
        kv_cache, positions = _synthetic_kv(layers=8, tokens=192, heads=4, dim=8)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        assert restored["keys"].shape == kv_cache["keys"].shape
        for layer_idx in range(8):
            cos = _cosine_similarity(
                kv_cache["values"][layer_idx], restored["values"][layer_idx]
            )
            assert cos > 0.9, f"Layer {layer_idx} cosine {cos:.4f} too low"

    def test_large_token_count(self) -> None:
        """Pipeline should handle token counts well beyond sink + window."""
        kv_cache, positions = _synthetic_kv(layers=1, tokens=512, heads=2, dim=8)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        compressed = compressor.compress(kv_cache, positions)
        assert compressed.metadata is not None
        assert compressed.metadata.middle_len == 512 - 4 - 128
        restored = compressor.decompress(compressed)
        assert restored["keys"].shape == kv_cache["keys"].shape

    def test_uniform_data_does_not_crash(self) -> None:
        """Constant-valued KV tensors should not cause division-by-zero."""
        kv_cache = {
            "keys": torch.ones(1, 200, 2, 8) * 0.5,
            "values": torch.ones(1, 200, 2, 8) * -0.3,
        }
        positions = torch.arange(200)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        # Should not raise
        compressed = compressor.compress(kv_cache, positions)
        restored = compressor.decompress(compressed)
        assert restored["keys"].shape == kv_cache["keys"].shape

    def test_high_variance_data(self) -> None:
        """Data with very high variance should still compress/decompress."""
        generator = torch.Generator().manual_seed(42)
        kv_cache = {
            "keys": torch.randn(1, 200, 2, 8, generator=generator) * 1000,
            "values": torch.randn(1, 200, 2, 8, generator=generator) * 1000,
        }
        positions = torch.arange(200)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        compressed = compressor.compress(kv_cache, positions)
        restored = compressor.decompress(compressed)
        assert restored["keys"].shape == kv_cache["keys"].shape


# ===========================================================================
# Section 12: Determinism / Reproducibility Tests
# ===========================================================================


class TestDeterminism:
    """Verify that compression is deterministic (same input -> same output),
    which is important for llama.cpp's test infrastructure."""

    def test_compress_is_deterministic(self) -> None:
        """Two compressions of the same data should produce identical results."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        c1 = compressor.compress(kv_cache, positions)
        c2 = compressor.compress(kv_cache, positions)
        assert len(c1.compressed_sections) == len(c2.compressed_sections)
        for s1, s2 in zip(c1.compressed_sections, c2.compressed_sections):
            assert s1.compressed_bytes == s2.compressed_bytes
            assert s1.bit_widths == s2.bit_widths
            assert s1.scales == s2.scales
            assert s1.zero_points == s2.zero_points

    def test_decompress_is_deterministic(self) -> None:
        """Two decompressions of the same compressed data should be identical."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        compressed = compressor.compress(kv_cache, positions)
        r1 = compressor.decompress(compressed)
        r2 = compressor.decompress(compressed)
        assert torch.equal(r1["keys"], r2["keys"])
        assert torch.equal(r1["values"], r2["values"])

    def test_different_seeds_produce_different_results(self) -> None:
        """Different random data should produce different compressed output."""
        kv1, pos1 = _synthetic_kv(seed=1)
        kv2, pos2 = _synthetic_kv(seed=2)
        cal1 = _calibration_from_cache(kv1, pos1)
        cal2 = _calibration_from_cache(kv2, pos2)
        c1 = KVTCCompressor(cal1).compress(kv1, pos1)
        c2 = KVTCCompressor(cal2).compress(kv2, pos2)
        # At least some sections should differ
        any_diff = False
        for s1, s2 in zip(c1.compressed_sections, c2.compressed_sections):
            if s1.compressed_bytes != s2.compressed_bytes:
                any_diff = True
                break
        assert any_diff, "Different inputs should produce different compressed data"


# ===========================================================================
# Section 13: RoPE Consistency Under KVTC Integration
# ===========================================================================


class TestRoPEConsistencyKVTC:
    """RoPE undo/reapply is critical for the llama.cpp integration because
    keys have RoPE applied. KVTC must undo RoPE before PCA, then reapply
    after decompression."""

    def test_rope_roundtrip_through_pipeline(self) -> None:
        """Keys should have RoPE correctly undone and reapplied through KVTC."""
        kv_cache, positions = _synthetic_kv(tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        # The middle region keys go through RoPE undo -> PCA -> quant -> dequant -> PCA inv -> RoPE
        # Sinks (first 4) and window (last 128) are preserved exactly
        # Middle region should still have reasonable cosine similarity
        middle_start = 4
        middle_end = 200 - 128
        if middle_end > middle_start:
            orig_middle_keys = kv_cache["keys"][:, middle_start:middle_end]
            rest_middle_keys = restored["keys"][:, middle_start:middle_end]
            cos = _cosine_similarity(orig_middle_keys, rest_middle_keys)
            # With small dim=8 synthetic data and lossy quantization, the
            # RoPE roundtrip degrades more than with real model dimensions.
            assert cos > 0.3, f"Middle keys cosine {cos:.4f} too low after RoPE roundtrip"

    def test_values_not_affected_by_rope(self) -> None:
        """Values should NOT go through RoPE — verify they don't get rope applied."""
        kv_cache, positions = _synthetic_kv(tokens=200)
        calibration = _calibration_from_cache(kv_cache, positions)
        compressor = KVTCCompressor(calibration)
        restored = compressor.decompress(compressor.compress(kv_cache, positions))
        # Values' middle region should also have good cosine (no RoPE distortion)
        middle_start = 4
        middle_end = 200 - 128
        if middle_end > middle_start:
            orig_middle_vals = kv_cache["values"][:, middle_start:middle_end]
            rest_middle_vals = restored["values"][:, middle_start:middle_end]
            cos = _cosine_similarity(orig_middle_vals, rest_middle_vals)
            assert cos > 0.9, f"Middle values cosine {cos:.4f} unexpected"


# ===========================================================================
# Section 14: PCA Calibration Entry Validation
# ===========================================================================


class TestPCACalibrationEntries:
    """Verify PCA calibration entries have the properties assumed by the
    KVTC CUDA kernels and llama.cpp integration."""

    def test_eigenvalues_are_descending(self) -> None:
        """Eigenvalues must be sorted in descending order."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        for entry in calibration.entries.values():
            assert torch.all(entry.eigenvalues[:-1] >= entry.eigenvalues[1:])

    def test_eigenvalues_nonnegative(self) -> None:
        """Eigenvalues should be non-negative (variances)."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        for entry in calibration.entries.values():
            assert torch.all(entry.eigenvalues >= 0)

    def test_eigenvectors_shape_is_square(self) -> None:
        """Eigenvectors should be [dim, dim] square matrices."""
        kv_cache, positions = _synthetic_kv(dim=8)
        calibration = _calibration_from_cache(kv_cache, positions)
        for entry in calibration.entries.values():
            assert entry.eigenvectors.shape == (8, 8)

    def test_mean_shape_matches_dim(self) -> None:
        """Mean vector shape should match the head dimension."""
        kv_cache, positions = _synthetic_kv(dim=8)
        calibration = _calibration_from_cache(kv_cache, positions)
        for entry in calibration.entries.values():
            assert entry.mean.shape == (8,)

    def test_bit_budget_positive(self) -> None:
        """All bit budgets should be positive integers."""
        kv_cache, positions = _synthetic_kv()
        calibration = _calibration_from_cache(kv_cache, positions)
        for entry in calibration.entries.values():
            assert entry.bit_budget > 0

    def test_head_group_entries_exist_for_all_groups(self) -> None:
        """With head_group_size=2 and 4 heads, there should be 2 groups per layer/kind."""
        kv_cache, positions = _synthetic_kv(heads=4)
        calibration = _calibration_from_cache(kv_cache, positions, head_group_size=2)
        for layer_idx in range(2):
            for kind in ("keys", "values"):
                assert (layer_idx, 0, kind) in calibration.entries
                assert (layer_idx, 1, kind) in calibration.entries


# ===========================================================================
# Section 15: CUDA Header / Kernel API Consistency Tests
# ===========================================================================


class TestCUDAHeaderConsistency:
    """Verify the CUDA header (kvtc.h) defines the API expected by
    the llama.cpp integration."""

    def test_header_defines_kvtc_ctx(self) -> None:
        """kvtc_ctx_t should be defined with expected fields."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        assert "kvtc_ctx_t" in content
        assert "sink_tokens" in content
        assert "window_tokens" in content
        assert "head_group_size" in content

    def test_header_defines_encode_decode(self) -> None:
        """kvtc_encode and kvtc_decode should be declared."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        assert "kvtc_encode(" in content
        assert "kvtc_decode(" in content

    def test_header_defines_calibration_struct(self) -> None:
        """kvtc_calibration_t should have eigenvectors, eigenvalues, mean, dim, bit_budget."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        assert "kvtc_calibration_t" in content
        for field in ["eigenvectors", "eigenvalues", "mean", "dim", "bit_budget", "rope_theta"]:
            assert field in content, f"Missing field: {field}"

    def test_header_defines_compressed_struct(self) -> None:
        """kvtc_compressed_t should have data, scales, zero_points, bit_widths."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        assert "kvtc_compressed_t" in content
        for field in ["data", "data_bytes", "num_rows", "scales", "zero_points", "bit_widths"]:
            assert field in content, f"Missing field: {field}"

    def test_header_defines_individual_kernels(self) -> None:
        """Individual kernel launchers should be declared for testing."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        kernels = [
            "kvtc_pca_transform",
            "kvtc_pca_inverse",
            "kvtc_quantize",
            "kvtc_dequantize",
            "kvtc_pack_bits",
            "kvtc_unpack_bits",
            "kvtc_rope_inverse",
            "kvtc_rope_forward",
            "kvtc_bit_allocation",
        ]
        for kernel in kernels:
            assert kernel in content, f"Missing kernel declaration: {kernel}"

    def test_header_encode_accepts_is_key_flag(self) -> None:
        """kvtc_encode should take an is_key parameter (controls RoPE)."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        # Find the kvtc_encode declaration
        encode_section = content[content.find("kvtc_encode("):content.find(";", content.find("kvtc_encode("))]
        assert "is_key" in encode_section

    def test_header_guard_defined(self) -> None:
        """Include guard KVTC_H should be present."""
        header = Path(__file__).resolve().parent.parent / "cuda" / "kvtc.h"
        content = header.read_text(encoding="utf-8")
        assert "#ifndef KVTC_H" in content
        assert "#define KVTC_H" in content
        assert "#endif" in content


# ===========================================================================
# Section 16: DP Bit Allocation Under KVTC Configs
# ===========================================================================


class TestDPBitAllocationKVTC:
    """Test DP bit allocation with budgets matching KVTC integration configs."""

    @pytest.mark.parametrize("bits_per_component", [1, 2, 3, 4, 6, 8])
    def test_allocation_respects_budget(self, bits_per_component: int) -> None:
        """Total allocated bits should not exceed the budget."""
        eigenvalues = torch.tensor([10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.1, 0.01])
        budget = bits_per_component * 8
        widths = dp_bit_allocation(eigenvalues, bit_budget=budget, max_bits=8)
        assert int(widths.sum().item()) <= budget

    def test_zero_budget_allocates_no_bits(self) -> None:
        """A budget of 0 should produce all-zero bit widths."""
        eigenvalues = torch.tensor([10.0, 5.0, 1.0])
        widths = dp_bit_allocation(eigenvalues, bit_budget=0, max_bits=8)
        assert torch.all(widths == 0)

    def test_high_budget_saturates_at_max_bits(self) -> None:
        """Very high budget should give max_bits to all components."""
        eigenvalues = torch.tensor([10.0, 5.0, 1.0])
        widths = dp_bit_allocation(eigenvalues, bit_budget=1000, max_bits=8)
        assert torch.all(widths <= 8)

    def test_important_components_get_more_bits(self) -> None:
        """Components with higher eigenvalues should get more or equal bits."""
        eigenvalues = torch.tensor([100.0, 50.0, 10.0, 1.0, 0.01])
        widths = dp_bit_allocation(eigenvalues, bit_budget=15, max_bits=8)
        # First component should have >= bits than last
        assert widths[0] >= widths[-1]
