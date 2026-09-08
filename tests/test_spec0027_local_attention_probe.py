# Copyright 2026 HiperMaximus
# ruff: noqa: S404, S603
"""Focused invariants for the WSI45630 local-softmax repair probe."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

KERNEL_ROOT = Path("kaggle/kernels/wsi45630_local_attention_probe")
SOURCE_PATH = KERNEL_ROOT / "run.py"
METADATA_PATH = KERNEL_ROOT / "kernel-metadata.json"
MAX_SOURCE_BYTES = 1_000_000
MIN_INPUT_GRADIENT_NORM = 1e-3
MIN_PARAMETER_GRADIENT_NORM = 1e-6
FAKE_ACCEPTED_VERSION = 2
type GraphResult = tuple[
    list[list[int]],
    list[list[bool]],
    list[list[int]],
    list[int],
]


class ProbeContract(Protocol):
    """Typed surface imported from the intentionally standalone kernel script."""

    TOKEN_WIDTH: int
    BACKENDS: tuple[str, ...]
    CHUNK_SIZES: tuple[int, ...]
    MAX_CORRECTNESS_RELATIVE_L2: float
    MIN_CORRECTNESS_COSINE: float
    build_graph: Callable[[Sequence[tuple[int, int]]], GraphResult]
    graph_sha256: Callable[
        [
            Sequence[tuple[int, int]],
            Sequence[Sequence[int]],
            Sequence[Sequence[bool]],
            Sequence[Sequence[int]],
        ],
        str,
    ]
    make_module: Callable[[object], torch.nn.Module]
    oracle_forward: Callable[..., torch.Tensor]
    comparison_metrics: Callable[
        [object, torch.Tensor, torch.Tensor],
        dict[str, object],
    ]
    correctness_gate: Callable[[dict[str, dict[str, object]]], bool]
    select_winner: Callable[
        [list[dict[str, object]]],
        dict[str, object] | None,
    ]


def _load_probe() -> ProbeContract:
    """Load the stdlib-safe kernel source so CPU tests exercise its real helpers.

    Returns:
        The imported one-off kernel module.

    """
    spec = importlib.util.spec_from_file_location("spec0028_probe", SOURCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast("ProbeContract", module)


def test_private_kernel_mounts_only_the_existing_coordinate_contract() -> None:
    """The probe must not gain latent, competition, model or test-data access."""
    metadata = cast("dict[str, object]", json.loads(METADATA_PATH.read_text()))
    assert metadata == {
        "id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
        "title": "eqvae WSI45630 local attention repair probe",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["maximusshtefan/eqvae-wsi45630-capacity-inputs"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def test_graph_uses_real_lattice_neighbours_and_preserves_holes() -> None:
    """Coordinate topology, not row adjacency, must define the dynamic kernel."""
    probe = _load_probe()
    coordinates = ((0, 0), (256, 0), (0, 256), (512, 512), (1280, 0))
    indices, valid, radial, degrees = probe.build_graph(coordinates)
    assert degrees == [4, 4, 4, 4, 1]
    assert [
        index for index, keep in zip(indices[0], valid[0], strict=True) if keep
    ] == [0, 2, 1, 3]
    assert [code for code, keep in zip(radial[0], valid[0], strict=True) if keep] == [
        0,
        1,
        1,
        5,
    ]
    assert all(
        index == -1
        for index, keep in zip(indices[4], valid[4], strict=True)
        if not keep
    )


def test_oracle_catches_mask_radial_and_nonzero_null_algebra_on_cpu() -> None:  # noqa: PLR0914, PLR0915
    """Independent oracle, explicit gather and SDPA must agree on hostile inputs."""
    probe = _load_probe()
    torch.manual_seed(27)  # pyright: ignore[reportUnknownMemberType]
    module_oracle = probe.make_module(torch)
    module_a = probe.make_module(torch)
    module_b = probe.make_module(torch)
    state = module_oracle.state_dict()
    state["relative_bias"] = torch.linspace(-0.3, 0.3, 36).reshape(6, 6)
    state["null_key"] = torch.linspace(-0.2, 0.2, 192).reshape(6, 32)
    state["null_bias"] = torch.linspace(-0.15, 0.15, 6)
    module_oracle.load_state_dict(state)
    module_a.load_state_dict(state)
    module_b.load_state_dict(state)
    coordinates = ((0, 0), (256, 0), (0, 256), (512, 512), (1280, 0))
    indices, valid, radial, _ = probe.build_graph(coordinates)
    graph = (
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(valid, dtype=torch.bool),
        torch.tensor(radial, dtype=torch.uint8),
    )
    x_oracle = torch.randn(5, probe.TOKEN_WIDTH, requires_grad=True)
    x_a = x_oracle.detach().clone().requires_grad_()
    x_b = x_oracle.detach().clone().requires_grad_()
    out_oracle = probe.oracle_forward(
        torch,
        module_oracle,
        x_oracle,
        *graph,
        len(coordinates),
    )
    out_a = cast(
        "torch.Tensor",
        module_a(x_a, *graph, backend="explicit", chunk_size=5),
    )
    out_b = cast(
        "torch.Tensor",
        module_b(x_b, *graph, backend="sdpa_auto", chunk_size=5),
    )
    upstream_phase = torch.arange(out_oracle.numel()).reshape_as(out_oracle)
    upstream = torch.cos(upstream_phase * 0.013)
    (out_oracle * upstream).sum().backward()  # pyright: ignore[reportUnknownMemberType]
    (out_a * upstream).sum().backward()  # pyright: ignore[reportUnknownMemberType]
    (out_b * upstream).sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert torch.allclose(out_oracle, out_a, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_oracle, out_b, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6)
    assert x_oracle.grad is not None
    assert x_a.grad is not None
    assert x_b.grad is not None
    assert (
        torch.linalg.vector_norm(x_oracle.grad)  # pyright: ignore[reportUnknownMemberType]
        > MIN_INPUT_GRADIENT_NORM
    )
    assert torch.allclose(x_oracle.grad, x_a.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(x_oracle.grad, x_b.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(x_a.grad, x_b.grad, rtol=1e-5, atol=1e-6)
    state_a = module_a.state_dict()
    null_key = cast("torch.Tensor", state_a["null_key"])
    null_bias = cast("torch.Tensor", state_a["null_bias"])
    assert torch.count_nonzero(null_key).item() == null_key.numel()
    assert torch.count_nonzero(null_bias).item() == null_bias.numel()
    for (name_oracle, parameter_oracle), (name_a, parameter_a), (
        name_b,
        parameter_b,
    ) in zip(
        module_oracle.named_parameters(),
        module_a.named_parameters(),
        module_b.named_parameters(),
        strict=True,
    ):
        assert name_oracle == name_a
        assert name_a == name_b
        assert parameter_oracle.grad is not None
        assert parameter_a.grad is not None
        assert parameter_b.grad is not None
        assert (
            torch.linalg.vector_norm(  # pyright: ignore[reportUnknownMemberType]
                parameter_oracle.grad,
            )
            > MIN_PARAMETER_GRADIENT_NORM
        )
        assert torch.allclose(
            parameter_oracle.grad,
            parameter_a.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        assert torch.allclose(
            parameter_oracle.grad,
            parameter_b.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        assert torch.allclose(parameter_a.grad, parameter_b.grad, rtol=1e-5, atol=1e-6)


def test_graph_hash_domain_separates_wsi_and_canonical_fields() -> None:
    """The graph identity must change with coordinates, masks, or sentinel bytes."""
    probe = _load_probe()
    coordinates = ((0, 0), (256, 0), (0, 256))
    indices, valid, radial, _ = probe.build_graph(coordinates)
    baseline = probe.graph_sha256(coordinates, indices, valid, radial)
    moved = ((256, 0), (512, 0), (256, 256))
    assert probe.graph_sha256(moved, indices, valid, radial) != baseline
    changed_valid = [row.copy() for row in valid]
    changed_valid[0][-1] = not changed_valid[0][-1]
    assert probe.graph_sha256(coordinates, indices, changed_valid, radial) != baseline
    changed_radial = [row.copy() for row in radial]
    changed_radial[0][-1] = 254
    assert probe.graph_sha256(coordinates, indices, valid, changed_radial) != baseline


def test_revised_gradient_gate_records_elementwise_failure_normwise_match() -> None:
    """Near-zero FP16 element drift is diagnostic while output allclose stays strict."""
    probe = _load_probe()
    expected = torch.cat((torch.zeros(1), torch.full((99,), 100.0)))
    actual = expected.clone()
    actual[0] = 0.01
    metric = probe.comparison_metrics(torch, actual, expected)
    assert metric["allclose"] is False
    assert metric["violation_count"] == 1
    assert cast("float", metric["relative_l2"]) < probe.MAX_CORRECTNESS_RELATIVE_L2
    assert cast("float", metric["cosine_similarity"]) >= probe.MIN_CORRECTNESS_COSINE
    assert probe.correctness_gate({"oracle:grad:test": metric}) is True
    assert probe.correctness_gate({"oracle:output": metric}) is False
    excessive_error = dict(metric)
    excessive_error["relative_l2"] = probe.MAX_CORRECTNESS_RELATIVE_L2 * 2
    assert probe.correctness_gate({"oracle:grad:test": excessive_error}) is False
    low_cosine = dict(metric)
    low_cosine["cosine_similarity"] = probe.MIN_CORRECTNESS_COSINE - 0.001
    assert probe.correctness_gate({"oracle:grad:test": low_cosine}) is False
    nonfinite = dict(metric)
    nonfinite["finite"] = False
    assert probe.correctness_gate({"oracle:grad:test": nonfinite}) is False


def test_candidate_matrix_and_winner_are_bounded() -> None:
    """The single run must compare only declared rows and never select failures."""
    probe = _load_probe()
    assert probe.BACKENDS == (
        "explicit",
        "sdpa_auto",
        "sdpa_efficient",
        "sdpa_flash",
    )
    assert probe.CHUNK_SIZES == (2048, 8192, 32595)
    rows: list[dict[str, object]] = [
        {"status": "failed", "backend": "explicit", "chunk_size": 2048, "mean_ms": 1.0},
        {
            "status": "ok",
            "backend": "sdpa_auto",
            "chunk_size": 8192,
            "mean_ms": 3.0,
            "finite_gradients": True,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
        },
        {
            "status": "ok",
            "backend": "sdpa_efficient",
            "chunk_size": 32595,
            "mean_ms": 2.0,
            "finite_gradients": True,
            "peak_allocated_bytes": 30,
            "peak_reserved_bytes": 40,
        },
    ]
    assert probe.select_winner(rows) == {
        "backend": "sdpa_efficient",
        "chunk_size": 32595,
        "mean_ms": 2.0,
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": 40,
    }


def test_push_guard_requires_the_exact_probe_confirmation() -> None:
    """Generic Kaggle permission must not authorize this one-use remote write."""
    environment = os.environ.copy()
    environment.update({
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
    })
    environment.pop("KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED", None)
    completed = subprocess.run(
        ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode != 0
    assert (
        "exact Spec 0028 local-attention repair confirmation/path required"
        in completed.stderr
    )


def test_consumed_probe_receipt_blocks_a_second_push() -> None:
    """A persistent generic confirmation cannot launch another kernel version."""
    receipt = Path(
        "runs/local/wsi45630_local_attention_repair_probe/push_receipt.json",
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    created = not receipt.exists()
    if created:
        receipt.write_text('{"authority_consumed":true}\n', encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED": "1",
    })
    try:
        completed = subprocess.run(
            ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    finally:
        if created:
            receipt.unlink()
    assert completed.returncode != 0
    assert "one-run authority was already consumed" in completed.stderr


def test_fixed_kernel_identity_cannot_fall_through_without_marker(
    tmp_path: Path,
) -> None:
    """The protected kernel ID must fail closed even when its marker is removed."""
    copied_kernel = tmp_path / "markerless"
    copied_kernel.mkdir()
    source = SOURCE_PATH.read_text(encoding="utf-8").replace(
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_READY = True",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_READY = False",
    )
    (copied_kernel / "run.py").write_text(source, encoding="utf-8")
    (copied_kernel / "kernel-metadata.json").write_bytes(METADATA_PATH.read_bytes())
    environment = os.environ.copy()
    environment.update({
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED": "1",
    })
    completed = subprocess.run(
        ["./scripts/kaggle_kernel.sh", "push", str(copied_kernel)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode != 0
    assert (
        "exact Spec 0028 local-attention repair confirmation/path required"
        in completed.stderr
    )


def test_corrupt_upload_snapshot_fails_before_remote_call(tmp_path: Path) -> None:
    """Snapshot drift after the canonical guard must consume no remote call."""
    receipt = Path(
        "runs/local/wsi45630_local_attention_repair_probe/push_receipt.json",
    )
    if receipt.exists():
        return
    fake_calls = tmp_path / "kaggle_calls"
    fake_cp = tmp_path / "cp"
    fake_cp.write_text(
        """#!/usr/bin/env bash
/bin/cp "$@" || exit 1
if [[ "$1" == */run.py ]]; then
  printf '\\n# corrupted snapshot\\n' >>"$2"
fi
""",
        encoding="utf-8",
    )
    fake_cp.chmod(0o755)
    fake_kaggle = tmp_path / "kaggle"
    fake_kaggle.write_text(
        f"#!/usr/bin/env bash\nprintf 'called\\n' >>'{fake_calls}'\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{tmp_path}:{environment['PATH']}",
        "KAGGLE_DISABLE_FRESH_OAUTH": "1",
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED": "1",
    })
    try:
        completed = subprocess.run(
            ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert completed.returncode != 0
        assert "failed to create exact Spec 0028 upload snapshot" in completed.stderr
        assert not fake_calls.exists()
        assert not receipt.exists()
    finally:
        if receipt.exists():
            receipt.unlink()


def test_probe_push_and_retrieval_routes_are_closed() -> None:
    """The one-shot route must verify acceptance and retrieve JSON plus logs."""
    wrapper = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert "Spec 0028 push forbids wait and all CLI overrides" in wrapper
    assert "Kaggle did not confirm an accepted local-attention probe version" in wrapper
    assert '"accepted_version": int(sys.argv[4])' in wrapper
    assert "output-local-attention-repair-probe)" in wrapper
    assert 'kaggle_api kernels logs "$versioned_kernel_id"' in wrapper
    assert "--file-pattern '^spec0028_local_attention_repair_probe[.]json$'" in wrapper
    assert 'mv "$stage_dir" "$local_attention_repair_probe_output_dir"' in wrapper
    assert '"artifact_sha256"' in wrapper
    assert '"log_sha256"' in wrapper


def test_one_shot_wrapper_binds_accepted_version_and_stages_evidence(
    tmp_path: Path,
) -> None:
    """A fake CLI proves acceptance parsing and exact-version retrieval behavior."""
    receipt = Path(
        "runs/local/wsi45630_local_attention_repair_probe/push_receipt.json",
    )
    if receipt.exists():
        payload = cast(
            "dict[str, object]",
            json.loads(receipt.read_text(encoding="utf-8")),
        )
        assert payload["authority_consumed"] is True
        assert isinstance(payload["accepted_version"], int)
        return
    fake_log = tmp_path / "calls.jsonl"
    fake_kaggle = tmp_path / "kaggle"
    fake_kaggle.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_KAGGLE_CALLS"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[:2] == ["kernels", "push"]:
    print("Kernel version 2 successfully pushed. Please check progress.")
elif args[:2] == ["kernels", "output"]:
    output = Path(args[args.index("-p") + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / "spec0028_local_attention_repair_probe.json").write_text(
        json.dumps({"spec": "0028", "status": "complete"}) + "\\n",
        encoding="utf-8",
    )
elif args[:2] == ["kernels", "logs"]:
    if os.environ.get("FAKE_EMPTY_LOG") != "1":
        print("complete fake Kaggle log")
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    output_dir = tmp_path / "retrieved"
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{tmp_path}:{environment['PATH']}",
        "FAKE_KAGGLE_CALLS": str(fake_log),
        "KAGGLE_DISABLE_FRESH_OAUTH": "1",
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED": "1",
        "KAGGLE_REMOTE_CONFIRMED": "1",
        "EQVAE_LOCAL_ATTENTION_REPAIR_PROBE_OUTPUT_DIR": str(output_dir),
    })
    try:
        pushed = subprocess.run(
            ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert pushed.returncode == 0, pushed.stderr
        push_payload = cast(
            "dict[str, object]",
            json.loads(receipt.read_text(encoding="utf-8")),
        )
        assert push_payload["accepted_version"] == FAKE_ACCEPTED_VERSION
        failed_output = tmp_path / "failed_retrieval"
        environment["FAKE_EMPTY_LOG"] = "1"
        environment["EQVAE_LOCAL_ATTENTION_REPAIR_PROBE_OUTPUT_DIR"] = str(
            failed_output,
        )
        failed_retrieval = subprocess.run(
            ["./scripts/kaggle_kernel.sh", "output-local-attention-repair-probe"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert failed_retrieval.returncode != 0
        assert not failed_output.exists()
        environment.pop("FAKE_EMPTY_LOG")
        environment["EQVAE_LOCAL_ATTENTION_REPAIR_PROBE_OUTPUT_DIR"] = str(output_dir)
        retrieved = subprocess.run(
            ["./scripts/kaggle_kernel.sh", "output-local-attention-repair-probe"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert retrieved.returncode == 0, retrieved.stderr
        retrieval = cast(
            "dict[str, object]",
            json.loads(
                (output_dir / "retrieval_receipt.json").read_text(encoding="utf-8"),
            ),
        )
        assert retrieval["accepted_version"] == FAKE_ACCEPTED_VERSION
        assert (output_dir / "kaggle.log").stat().st_size > 0
        calls = [json.loads(line) for line in fake_log.read_text().splitlines()]
        assert calls == [
            ["kernels", "push", "-p", calls[0][3]],
            [
                "kernels",
                "output",
                "maximusshtefan/eqvae-wsi45630-local-attention-repair-probe/2",
                "-p",
                calls[1][4],
                "--file-pattern",
                "^spec0028_local_attention_repair_probe[.]json$",
            ],
            [
                "kernels",
                "logs",
                "maximusshtefan/eqvae-wsi45630-local-attention-repair-probe/2",
            ],
            [
                "kernels",
                "output",
                "maximusshtefan/eqvae-wsi45630-local-attention-repair-probe/2",
                "-p",
                calls[3][4],
                "--file-pattern",
                "^spec0028_local_attention_repair_probe[.]json$",
            ],
            [
                "kernels",
                "logs",
                "maximusshtefan/eqvae-wsi45630-local-attention-repair-probe/2",
            ],
        ]
        assert "spec0028_upload." in calls[0][3]
        assert "failed_retrieval.staging." in calls[1][4]
        assert "retrieved.staging." in calls[3][4]
    finally:
        if receipt.exists():
            receipt.unlink()


def test_ambiguous_or_wrong_version_consumes_the_attempt(tmp_path: Path) -> None:
    """No retry is possible after an ambiguous reply or an unexpected version."""
    receipt = Path(
        "runs/local/wsi45630_local_attention_repair_probe/push_receipt.json",
    )
    if receipt.exists():
        return
    fake_kaggle = tmp_path / "kaggle"
    fake_kaggle.write_text(
        """#!/usr/bin/env python3
import os

if os.environ["FAKE_REPLY"] == "wrong":
    print("Kernel version 3 successfully pushed. Please check progress.")
else:
    print("Upload accepted, but the version response was truncated.")
""",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{tmp_path}:{environment['PATH']}",
        "KAGGLE_DISABLE_FRESH_OAUTH": "1",
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED": "1",
    })
    try:
        for reply, expected_version in (("ambiguous", None), ("wrong", 3)):
            environment["FAKE_REPLY"] = reply
            completed = subprocess.run(
                ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            assert completed.returncode != 0
            payload = cast(
                "dict[str, object]",
                json.loads(receipt.read_text(encoding="utf-8")),
            )
            assert payload["authority_consumed"] is True
            assert payload["accepted_version"] == expected_version
            blocked_retry = subprocess.run(
                ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            assert blocked_retry.returncode != 0
            assert "one-run authority was already consumed" in blocked_retry.stderr
            receipt.unlink()
    finally:
        if receipt.exists():
            receipt.unlink()


def test_concurrent_pushes_share_one_atomic_attempt_claim(tmp_path: Path) -> None:
    """Two simultaneous wrappers may make at most one remote push call."""
    receipt = Path(
        "runs/local/wsi45630_local_attention_repair_probe/push_receipt.json",
    )
    if receipt.exists():
        return
    fake_calls = tmp_path / "calls.jsonl"
    fake_kaggle = tmp_path / "kaggle"
    fake_kaggle.write_text(
        """#!/usr/bin/env python3
import os
import time
from pathlib import Path

with Path(os.environ["FAKE_KAGGLE_CALLS"]).open("a", encoding="utf-8") as handle:
    handle.write("push\\n")
time.sleep(0.2)
print("Kernel version 2 successfully pushed. Please check progress.")
""",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{tmp_path}:{environment['PATH']}",
        "FAKE_KAGGLE_CALLS": str(fake_calls),
        "KAGGLE_DISABLE_FRESH_OAUTH": "1",
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
        "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED": "1",
    })
    processes = [
        subprocess.Popen(
            ["./scripts/kaggle_kernel.sh", "push", str(KERNEL_ROOT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(2)
    ]
    try:
        results = [process.communicate(timeout=10) for process in processes]
        assert sorted(process.returncode for process in processes) == [0, 1]
        assert fake_calls.read_text(encoding="utf-8").splitlines() == ["push"]
        assert any(
            "one-run authority was already consumed" in stderr
            or "File exists" in stderr
            for _, stderr in results
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        if receipt.exists():
            receipt.unlink()


def test_source_is_small_compilable_and_pins_remote_inputs() -> None:
    """The upload must remain auditable and fail closed on coordinate drift."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert SOURCE_PATH.stat().st_size < MAX_SOURCE_BYTES
    compile(source, str(SOURCE_PATH), "exec")
    assert "99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660" in source
    assert "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c" in source
    assert "N x N" not in source
