# Copyright 2026 HiperMaximus
"""Packaging guards for the private Spec 0015 readiness kernel."""
# pyright: reportAny=false

from __future__ import annotations

import base64
import io
import json
import re
import runpy
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable


def test_readiness_kernel_embeds_fixed_source_and_no_dataset(tmp_path: Path) -> None:
    """The uploadable script is self-contained and carries no data attachment."""
    repository = Path(__file__).resolve().parents[1]
    kernel_dir = repository / "kaggle/kernels/so2_runtime_readiness"
    output = tmp_path / "run.py"
    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repository / "scripts/build_kaggle_embedded_kernel.py"),
            "--repo-root",
            str(repository),
            "--kernel-dir",
            str(kernel_dir),
            "--output-run",
            str(output),
            "--ready-marker",
            "KAGGLE_SO2_RUNTIME_READINESS_READY = True",
            "--allow-dirty",
        ),
        cwd=repository,
        check=True,
    )
    metadata = cast(
        "dict[str, object]",
        json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")),
    )
    assert metadata["id"] == "maximusshtefan/eqvae-so2-runtime-readiness"
    for field in (
        "dataset_sources",
        "competition_sources",
        "kernel_sources",
        "model_sources",
    ):
        assert metadata[field] == []

    match = re.search(
        r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
        output.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert match is not None
    with zipfile.ZipFile(
        io.BytesIO(base64.b64decode(match.group("payload"))),
    ) as archive:
        names = set(archive.namelist())
        source = archive.read(
            "src/eqvae/benchmarking/so2_runtime_readiness.py",
        ).decode("utf-8")
        config = json.loads(
            archive.read(
                "configs/spec0015/so2_vae_selected_runtime_readiness.json",
            ),
        )
    assert config["data"]["dataset_sources"] == []
    assert config["readiness"]["per_device_batch_size"] == 1
    assert "src/eqvae/models/so2_vae.py" in names
    assert "PER_DEVICE_BATCH: Final = 1" in source
    assert "GATE_ROW_COUNT: Final = 68" in source
    assert "make_fastpath_step_fn" in source
    assert "run_fastpath_optimizer_step_with_metrics" in source
    assert "full_training_authorized" in source


def test_readiness_kernel_has_dedicated_guard_and_continuation_commands() -> None:
    """The kernel cannot fall through to a training or legacy probe guard."""
    repository = Path(__file__).resolve().parents[1]
    script = (repository / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    for required in (
        "guard_so2_runtime_readiness_push_ready",
        "preflight-so2-runtime-readiness",
        "status-so2-runtime-readiness",
        "wait-so2-runtime-readiness",
        "output-so2-runtime-readiness",
        "KAGGLE_SO2_RUNTIME_READINESS_READY = True",
        "do not attach a dataset to SO(2) readiness",
    ):
        assert required in script


def _template_namespace() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    return runpy.run_path(
        str(repository / "kaggle/kernels/so2_runtime_readiness/run_template.py"),
        run_name="spec0015_template_test",
    )


def test_download_validator_rejects_rank_metric_aggregate_forgery() -> None:
    """Rank-one instability cannot hide behind forged rank-zero aggregates."""
    validate = cast(
        "Callable[[dict[str, object], dict[str, object]], list[str]]",
        _template_namespace()["_rank_metric_errors"],
    )
    metrics: list[dict[str, object]] = [
        {
            "rank": rank,
            "compile_startup_seconds": 2.0 + rank,
            "settled_step_ms": [10.0, 11.0, 12.0],
            "peak_allocated_mib": 1000.0 + rank,
            "peak_reserved_mib": 1200.0 + rank,
            "total_device_memory_mib": 15000.0,
            "reserved_headroom_fraction": (15000.0 - (1200.0 + rank)) / 15000.0,
            "amp_step_skipped_count": 0,
            "post_settle_graph_break_count": 0,
            "post_settle_recompile_count": 0,
            "finite_losses": True,
            "finite_parameters": True,
        }
        for rank in (0, 1)
    ]
    compiled: dict[str, object] = {
        "amp_step_skipped_count": 0,
        "post_settle_graph_break_count": 0,
        "post_settle_recompile_count": 0,
        "finite_losses": True,
        "finite_parameters": True,
        "peak_allocated_mib_rank_max": 1001.0,
        "peak_reserved_mib_rank_max": 1201.0,
        "reserved_vram_headroom_fraction_rank_min": (15000.0 - 1201.0) / 15000.0,
        "compile_startup_seconds_rank_max": 3.0,
        "diagnostic_settled_step_ms_p50": 11.0,
        "diagnostic_settled_step_ms_rank_samples": [
            [10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0],
        ],
    }
    payload: dict[str, object] = {"rank_metrics": metrics}
    assert validate(payload, compiled) == []
    metrics[1]["post_settle_graph_break_count"] = 1
    assert validate(payload, compiled)


def test_download_validator_pins_proof_bodies_and_gate_identities() -> None:
    """Pass labels cannot replace concrete DDP/optimizer/model evidence."""
    namespace = _template_namespace()
    validate = cast(
        "Callable[[dict[str, object]], list[str]]",
        namespace["_proof_body_errors"],
    )
    expected_ddp = {
        "python_reducer": True,
        "static_graph": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "bucket_bytes_cap": 50 * 1024 * 1024,
        "optimize_ddp": "python_reducer",
        "compiled_autograd": True,
        "reorder_compute_comm_overlap": True,
        "optimizer_fused": True,
    }
    payload: dict[str, object] = {
        "master_dtype_proof": {
            "status": "pass",
            "parameter_dtypes": ["torch.float32"],
            "buffer_dtypes": ["torch.float32"],
            "field_norm_count": 40,
            "radial_gate_count": 34,
            "norm_and_radial_math_dtype": "float32",
        },
        "pre_compile_buffer_sync": {
            "status": "pass",
            "checked_before_ddp_and_compile": True,
            "buffer_count": 54,
            "max_abs_difference": 0.0,
            "worst_buffer": "",
        },
        "ddp_runtime_readback": {
            "status": "pass",
            "requested": expected_ddp,
            "effective": expected_ddp,
        },
        "optimizer_policy": {
            "status": "pass",
            "all_parameters_covered_once": True,
            "coefficient_parameter_count": 1_172_304,
            "gate_parameter_count": 4_096,
            "coefficient_weight_decay": 1e-5,
            "gate_weight_decay": 0.0,
            "base_learning_rate": 0.00014433756729740645,
            "gate_learning_rate": 0.00007216878364870322,
            "fused_requested": True,
        },
        "gradient_mean_reference": {
            "status": "pass",
            "parameter": "output_head.bias",
            "local_pre_reduction_gradients_differ": True,
            "reduced_gradient_max_abs_error": 0.0,
        },
        "parameter_sync": {
            "status": "pass",
            "max_abs_difference": 0.0,
            "worst_parameter": "",
        },
    }
    assert validate(payload) == []
    cast("dict[str, object]", payload["optimizer_policy"])["gate_parameter_count"] = 0
    assert validate(payload) == ["optimizer policy proof body must be exact"]
    modules = cast("set[str]", namespace["EXPECTED_GATE_MODULES"])
    assert len(modules) == namespace["RADIAL_GATE_COUNT"]
    assert "stem_gate" in modules
    assert "decoder_blocks.7.output_gate" in modules
