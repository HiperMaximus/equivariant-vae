# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportUnnecessaryCast=false
"""Regression tests for the guarded Spec 0051 kernel wrapper."""

from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from eqvae.evaluation.decoded_transform import (
    EXACT_D4_NAMES,
    paired_bootstrap_median_difference,
)

EXPECTED_PATCHES = 25
EXPECTED_DENSE_NONIDENTITY_ANGLES = 71
EXPECTED_WSI_CLUSTERS = 16


def test_decision_engine_supports_and_propagates_unresolved_state() -> None:
    """Decision logic honors valid ranks, joint H3, artifacts, and both models."""
    namespace = _template_namespace()
    decide = cast("Any", namespace["_decisions"])
    branches = {
        "normal_vae": _synthetic_branch(action=0.8, canonical=0.8),
        "so2_vae": _synthetic_branch(action=0.4, canonical=0.4),
    }
    decisions = decide(branches, np=__import__("numpy"))
    assert decisions["H1_decoded_rotational_action"]["status"] == "supported"
    assert decisions["H2_decoded_rotational_canonicalization"]["status"] == "supported"
    assert decisions["H3_decoded_residual_suppression"]["status"] == "supported"
    assert (
        decisions["H4_reflection_robustness"]["flip_h"]["interpretation"]
        == "generic_both_models"
    )

    for row in branches["so2_vae"]["per_patch"]:
        row["action_nondegenerate"] = False
        row["canonical_nondegenerate"] = False
    unresolved = decide(branches, np=__import__("numpy"))
    assert (
        unresolved["H1_decoded_rotational_action"]["status"]
        == "unresolved_insufficient_pose_signal"
    )
    assert (
        unresolved["H2_decoded_rotational_canonicalization"]["status"]
        == "unresolved_insufficient_pose_signal"
    )
    assert (
        unresolved["H3_decoded_residual_suppression"]["status"]
        == "unresolved_inherited_insufficient_pose_signal"
    )


def test_exact_quarter_requires_its_own_nondegenerate_signal() -> None:
    """Dense validity cannot make a degenerate exact-quarter control pass."""
    namespace = _template_namespace()
    decide = cast("Any", namespace["_decisions"])
    branches = {
        "normal_vae": _synthetic_branch(action=0.8, canonical=0.8),
        "so2_vae": _synthetic_branch(action=0.4, canonical=0.4),
    }
    exact = branches["so2_vae"]["exact_quarter_aggregate"]
    exact["joint_valid_per_patch"] = [False] * EXPECTED_PATCHES
    exact["joint_valid_count"] = 0
    decisions = decide(branches, np=__import__("numpy"))
    assert (
        decisions["H1_decoded_rotational_action"]["status"]
        == "unresolved_insufficient_exact_quarter_pose_signal"
    )
    assert (
        decisions["H2_decoded_rotational_canonicalization"]["status"]
        == "unresolved_insufficient_exact_quarter_pose_signal"
    )
    assert (
        decisions["H3_decoded_residual_suppression"]["status"]
        == "unresolved_inherited_insufficient_pose_signal"
    )


def test_h4_does_not_call_support_exclusive_when_peer_is_unresolved() -> None:
    """Aggregate H4 wording preserves a non-assessable peer model."""
    namespace = _template_namespace()
    decide = cast("Any", namespace["_decisions"])
    branches = {
        "normal_vae": _synthetic_branch(action=0.8, canonical=0.8),
        "so2_vae": _synthetic_branch(action=0.4, canonical=0.4),
    }
    branches["normal_vae"]["exact_d4"]["flip_h"]["joint_valid_count"] = 0
    decisions = decide(branches, np=__import__("numpy"))
    assert (
        decisions["H4_reflection_robustness"]["flip_h"]["interpretation"]
        == "so2_supported_normal_unresolved"
    )


def test_d4_model_comparison_excludes_any_model_degenerate_rank() -> None:
    """D4 support comparisons use the intersection valid for both models."""
    namespace = _template_namespace()
    compare = cast("Any", namespace["_comparisons"])
    normal = _synthetic_branch(action=0.8, canonical=0.8)
    so2 = _synthetic_branch(action=0.4, canonical=0.4)
    normal["exact_d4"]["flip_h"]["joint_valid_per_patch"][0] = False
    normal["exact_d4"]["flip_h"]["action_ratio_per_patch"][0] = 1_000.0
    output = compare(
        {"normal_vae": normal, "so2_vae": so2},
        cluster_ids=[str(index) for index in range(EXPECTED_PATCHES)],
        paired_bootstrap_median_difference=paired_bootstrap_median_difference,
        np=__import__("numpy"),
    )
    subset = output["exact_d4"]["flip_h"]["action_ratio_per_patch"][
        "supporting_joint_valid_subset"
    ]
    expected_median = 0.4
    assert subset["count"] == EXPECTED_PATCHES - 1
    assert 0 not in subset["ranks"]
    assert subset["normal_median"] == pytest.approx(expected_median)


def test_ratio_renderer_uses_its_nonidentity_angle_axis(tmp_path: Path) -> None:
    """The plot consumes 71 ratios against 71 angles rather than all 72 angles."""
    namespace = _template_namespace()
    render = cast("Any", namespace["_render_angle_series"])
    render.__globals__["FIGURE_ROOT"] = tmp_path
    summary = {
        "angles_degrees": list(range(5, 360, 5)),
        "median": [0.5] * EXPECTED_DENSE_NONIDENTITY_ANGLES,
        "q1": [0.4] * EXPECTED_DENSE_NONIDENTITY_ANGLES,
        "q3": [0.6] * EXPECTED_DENSE_NONIDENTITY_ANGLES,
    }
    branches = {
        branch: {"ratio_by_angle": {"action": summary, "canonical": summary}}
        for branch in ("normal_vae", "so2_vae")
    }
    fake_plt = MagicMock()
    fake_figure = MagicMock()
    fake_figure.savefig.side_effect = _touch_figure
    fake_plt.subplots.return_value = (fake_figure, [MagicMock(), MagicMock()])
    render(branches, plt=fake_plt, np=__import__("numpy"))
    assert (tmp_path / "02-ratio-by-angle.png").is_file()


def test_contract_hashes_and_shell_claim_are_bound_correctly() -> None:
    """Spec/contract bytes and the shell one-shot claim share one live binding."""
    root = Path(__file__).resolve().parents[1]
    spec = root / "docs/specs/0051-decoded-latent-transform-consistency.md"
    contract = root / "docs/data/spec0051_decoded_transform_contract.json"
    template = (
        root / "kaggle/kernels/decoded_latent_transform/run_template.py"
    ).read_text(encoding="utf-8")
    shell = (root / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    spec_hash = hashlib.sha256(spec.read_bytes()).hexdigest()
    contract_hash = hashlib.sha256(contract.read_bytes()).hexdigest()
    for source in (template, shell):
        assert spec_hash in source
        assert contract_hash in source
    assert "PYSPEC0050CLAIM\n}\n\nclaim_decoded_transform_push()" in shell


def test_shell_guard_claims_once_before_decoded_transform_push() -> None:
    """The exact confirmations and atomic claim precede remote mutation."""
    root = Path(__file__).resolve().parents[1]
    shell = (root / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    for marker in (
        "KAGGLE_DECODED_LATENT_TRANSFORM_CONFIRMED",
        "KAGGLE_FULL_DATASET_CONFIRMED",
        '[[ -e "$decoded_transform_claim" ]]',
        "runs/local/kaggle_launches/maximshtefan/eqvae-decoded-latent-transform/v*.json",
        'with claim.open("x", encoding="utf-8") as handle:',
        "Spec 0051 push forbids wait and all CLI overrides",
    ):
        assert marker in shell
    claim_call = shell.index(
        'claim_decoded_transform_push "$kernel_dir" "$upload_kernel_dir" "$actor"',
    )
    remote_push = shell.index(
        'kaggle_api kernels push -p "$upload_kernel_dir"',
        claim_call,
    )
    assert claim_call < remote_push


def test_embedded_kernel_build_is_byte_reproducible(tmp_path: Path) -> None:
    """Two unchanged builds produce byte-identical upload scripts."""
    root = Path(__file__).resolve().parents[1]
    outputs = (tmp_path / "first.py", tmp_path / "second.py")
    for output in outputs:
        subprocess.run(  # noqa: S603
            (
                sys.executable,
                "scripts/build_kaggle_embedded_kernel.py",
                "--kernel-dir",
                "kaggle/kernels/decoded_latent_transform",
                "--output-run",
                str(output),
                "--ready-marker",
                "KAGGLE_DECODED_LATENT_TRANSFORM_READY = True",
                "--allow-dirty",
            ),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_payload_zip_is_independent_of_source_mtime(tmp_path: Path) -> None:
    """Every embedded member uses deterministic ZIP metadata across checkouts."""
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(root / "scripts/build_kaggle_embedded_kernel.py"),
        run_name="spec0051_builder_test",
    )
    build_zip = cast("Any", namespace["_payload_zip_bytes"])
    source = tmp_path / "source.txt"
    source.write_text("fixed bytes\n", encoding="utf-8")

    def fixed_files(
        _repo_root: Path,
        _kernel_dir: Path,
    ) -> tuple[tuple[Path, str], ...]:
        return ((source, "source.txt"),)

    build_zip.__globals__["_payload_files"] = fixed_files
    os.utime(source, (1_000_000_000, 1_000_000_000))
    first = build_zip(tmp_path, b"{}", tmp_path)
    os.utime(source, (1_700_000_000, 1_700_000_000))
    second = build_zip(tmp_path, b"{}", tmp_path)
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert {info.date_time for info in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0),
        }


def test_contract_declares_complete_ordered_d4_and_sixteen_wsies() -> None:
    """The machine contract matches the exact code order and cluster count."""
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "docs/data/spec0051_decoded_transform_contract.json").read_text(
            encoding="utf-8",
        ),
    )
    selector = json.loads(
        (
            root / "runs/kaggle/fixed25_selector/fixed_25_validation_patches.json"
        ).read_text(encoding="utf-8"),
    )

    assert tuple(contract["exact_d4"]["elements"]) == EXACT_D4_NAMES
    assert (
        len({row["wsi_id"] for row in selector["selectors"]}) == EXPECTED_WSI_CLUSTERS
    )


def _template_namespace() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(
        str(root / "kaggle/kernels/decoded_latent_transform/run_template.py"),
        run_name="spec0051_test_template",
    )


def _touch_figure(path: Path, **_kwargs: object) -> None:
    path.touch()


def _synthetic_branch(*, action: float, canonical: float) -> dict[str, Any]:
    rows = [
        {
            "action_ratio": action,
            "canonical_ratio": canonical,
            "action_nondegenerate": True,
            "canonical_nondegenerate": True,
            "action_ssim_gain": 0.1,
            "canonical_ssim_gain": 0.1,
            "action_excess_out_of_range_fraction": 0.0,
            "canonical_excess_out_of_range_fraction": 0.0,
            "action_excess_overshoot": 0.0,
            "canonical_excess_overshoot": 0.0,
            "action_gradient_error_ratio": 0.5,
            "canonical_gradient_error_ratio": 0.5,
            "raw_latent_canonical_ratio": 0.9,
            "input_commutation_ratio": action,
            "action_rms_disk": action,
            "action_mae_disk": action,
            "action_rms_full": action,
            "action_mae_full": action,
            "canonical_rms_disk": canonical,
            "canonical_mae_disk": canonical,
            "canonical_rms_full": canonical,
            "canonical_mae_full": canonical,
            "wrong_sign_action_ratio": 1.0,
            "wrong_sign_canonical_ratio": 1.0,
            "output_control_canonical_ratio": 1.0,
        }
        for _ in range(EXPECTED_PATCHES)
    ]
    exact = {
        name: {
            "joint_valid_count": EXPECTED_PATCHES,
            "input_commutation_ratio_median_valid": 0.4,
            "input_commutation_success_fraction_valid": 1.0,
            "action_ratio_median_valid": 0.4,
            "action_success_fraction_valid": 1.0,
            "canonical_ratio_median_valid": 0.4,
            "canonical_success_fraction_valid": 1.0,
            "joint_valid_per_patch": [True] * EXPECTED_PATCHES,
            "action_ratio_per_patch": [0.4] * EXPECTED_PATCHES,
            "input_commutation_ratio_per_patch": [0.4] * EXPECTED_PATCHES,
            "canonical_ratio_per_patch": [0.4] * EXPECTED_PATCHES,
            "raw_latent_ratio_per_patch": [0.9] * EXPECTED_PATCHES,
        }
        for name in ("flip_h", "flip_diag", "flip_v", "flip_anti_diag")
    }
    return {
        "per_patch": rows,
        "exact_quarter_aggregate": {
            "action_ratio_per_patch": [0.4] * EXPECTED_PATCHES,
            "canonical_ratio_per_patch": [0.4] * EXPECTED_PATCHES,
            "joint_valid_per_patch": [True] * EXPECTED_PATCHES,
            "joint_valid_count": EXPECTED_PATCHES,
        },
        "exact_d4": exact,
    }
