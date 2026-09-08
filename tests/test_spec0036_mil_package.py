# Copyright 2026 HiperMaximus
# pyright: reportAny=false, reportIndexIssue=false, reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: PLR0913, PLR0915, PLR2004, SLF001
"""Package and remote-boundary contracts for Spec 0036 MIL training."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from scripts import build_ubc_mil_training as builder

from eqvae.data.supervised_latents import LogicalPointer, WSIInstance
from eqvae.kaggle_resources import create_portable_kernel_snapshot
from eqvae.models.local_global_mil import build_local_attention_graph
from eqvae.training.mil_training import BranchNumericalError

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType


@dataclass(frozen=True)
class BuiltPackage:
    """One immutable temporary package shared by the expensive focused checks."""

    root: Path
    contract: dict[str, object]


@pytest.fixture(scope="module")
def built_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[BuiltPackage]:
    """Build once because all 129 real graph hashes are one shared authority.

    Yields:
        The immutable package path and parsed contract.

    """
    output = tmp_path_factory.mktemp("spec0036_package") / "package"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(builder, "DEFAULT_ROOT", output)
    try:
        yield BuiltPackage(root=output, contract=builder.build(actor="researcher"))
    finally:
        monkeypatch.undo()


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spec0036_runtime", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _instances(coordinates: tuple[tuple[int, int], ...]) -> tuple[WSIInstance, ...]:
    return tuple(
        WSIInstance(
            instance_row=index,
            atlas_row_index=100 + index,
            wsi_id=17,
            x=x,
            y=y,
            diagnosis_label="HGSC",
            diagnosis_index=2,
            split="train",
            pointer=LogicalPointer(part=1, file_index=index),
        )
        for index, (x, y) in enumerate(coordinates)
    )


def test_package_stages_only_the_exact_development_view_and_physical_sources(
    built_package: BuiltPackage,
) -> None:
    """The private input exposes development pointers, never sealed-test metadata."""
    bundle = built_package.root / "bundle"
    staged = bundle / "development"
    assert {path.name for path in staged.iterdir()} == set(builder.DEVELOPMENT_FILES)
    assert all(path.is_file() and not path.is_symlink() for path in staged.iterdir())
    for name, expected_hash in builder.DEVELOPMENT_FILE_SHA256.items():
        source = builder.ROOT / builder.DEV_ROOT / name
        target = staged / name
        assert target.read_bytes() == source.read_bytes()
        assert builder._sha256(target) == expected_hash
    assert not (bundle / "sealed_test").exists()

    physical_sources = cast(
        "list[dict[str, str]]",
        built_package.contract["physical_sources"],
    )
    assert len(physical_sources) == 30
    assert physical_sources == builder._physical_sources(
        builder.ROOT / builder.DEV_ROOT / "physical_parts.csv",
    )
    assert (
        tuple(
            dict.fromkeys(row["kaggle_source"] for row in physical_sources),
        )
        == builder.KERNEL_SOURCES
    )
    assert all("/" in row["kaggle_source"] for row in physical_sources)
    assert all(len(row["binary_sha256"]) == 64 for row in physical_sources)
    assert all(len(row["sidecar_sha256"]) == 64 for row in physical_sources)


def test_package_is_actor_owned_while_every_upstream_owner_is_preserved(
    built_package: BuiltPackage,
    tmp_path: Path,
) -> None:
    """Only new resources follow the actor; physical producers retain provenance."""
    contract = built_package.contract
    assert contract["visibility"] == "private"
    assert contract["dataset_actor"] == "researcher"
    assert contract["dataset_reference"] == (
        "researcher/eqvae-local-global-mil-training-inputs-v3"
    )
    dataset_metadata = json.loads(
        (built_package.root / "bundle/dataset-metadata.json").read_text(
            encoding="utf-8",
        ),
    )
    assert dataset_metadata["id"] == contract["dataset_reference"]

    kernel = built_package.root / "kernel"
    metadata = json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] == "true"
    assert metadata["dataset_sources"] == [contract["dataset_reference"]]
    assert metadata["kernel_sources"] == list(builder.KERNEL_SOURCES)
    portable = tmp_path / "portable"
    snapshot = create_portable_kernel_snapshot(kernel, portable, actor="professor")
    portable_metadata = json.loads(
        (portable / "kernel-metadata.json").read_text(encoding="utf-8"),
    )
    assert portable_metadata["id"] == "professor/eqvae-local-global-mil-training"
    assert portable_metadata["dataset_sources"] == [contract["dataset_reference"]]
    assert snapshot.source_locators["kernel_sources"] == list(builder.KERNEL_SOURCES)


def test_contract_binds_config_state_sources_and_all_shared_graphs(
    built_package: BuiltPackage,
) -> None:
    """Every executable and geometry authority is hash-bound before latent reads."""
    contract = built_package.contract
    assert contract["schema_version"] == "spec0036.mil_training_input.v1"
    assert contract["spec_sha256"] == builder._sha256(builder.ROOT / builder.SPEC_PATH)
    assert contract["development_contract_sha256"] == builder.DEV_CONTRACT_SHA256
    assert contract["training_config"] == builder._training_config()
    assert contract["training_config_sha256"] == builder._json_sha256(
        builder._training_config(),
    )
    assert contract["model"] == {
        "source": builder.MODEL_PATH.as_posix(),
        "sha256": builder._sha256(builder.ROOT / builder.MODEL_PATH),
        "parameter_count": 1_513_055,
    }
    assert contract["candidate"] == {
        "source": builder.CANDIDATE_PATH.as_posix(),
        "sha256": builder._sha256(builder.ROOT / builder.CANDIDATE_PATH),
        "backend": "whole_bag_fixed25_inductor",
    }
    state_path = built_package.root / "bundle" / builder.INITIAL_STATE_NAME
    state = cast(
        "dict[str, torch.Tensor]",
        torch.load(state_path, map_location="cpu", weights_only=True),
    )
    state_contract = cast("dict[str, object]", contract["initial_state"])
    assert state_contract["file_sha256"] == builder._sha256(state_path)
    assert state_contract["state_sha256"] == builder._state_dict_sha256(state)
    assert sum(tensor.numel() for tensor in state.values()) == 1_513_055

    graphs = cast("dict[str, dict[str, str]]", contract["graph_identities"])
    assert {split: len(items) for split, items in graphs.items()} == {
        "train": 106,
        "validation": 23,
    }
    assert len({wsi for values in graphs.values() for wsi in values}) == 129
    assert all(
        len(identity) == 64
        for values in graphs.values()
        for identity in values.values()
    )

    source_files = {
        name
        for name in cast("dict[str, object]", contract["files"])
        if name.startswith("src/")
    }
    expected_sources = {
        f"src/{path.relative_to(builder.ROOT / 'src').as_posix()}"
        for path in builder._runtime_source_paths(builder.ROOT)
    }
    assert source_files == expected_sources
    assert (
        not {f"src/{path.as_posix()}" for path in builder.SOURCE_EXCLUDES}
        & source_files
    )


def test_verified_input_receipt_stays_outside_immutable_package() -> None:
    """Remote verification evidence must not invalidate the package allow-list."""
    launcher = (builder.ROOT / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert "mil_training_authority_root=" in launcher
    assert "runs/local/ubc_ocean_mil_training_authority" in launcher
    assert "$mil_training_authority_root/input_dataset_v3_receipt.json" in launcher


def test_fast_graph_precomputation_matches_the_canonical_model_identity() -> None:
    """The optimized package builder must encode the exact Spec 0026 graph arrays."""
    coordinates = ((256, 256), (0, 0), (512, 0), (0, 512), (512, 512), (1024, 1024))
    expected = build_local_attention_graph(
        _instances(coordinates),
        expected_instance_count=len(coordinates),
    ).identity_sha256
    assert (
        builder._graph_identity_from_coordinates(
            wsi_id=17,
            coordinates=coordinates,
        )
        == expected
    )


def test_build_is_immutable_and_upload_members_are_exactly_stored(
    built_package: BuiltPackage,
) -> None:
    """An existing package cannot be replaced and its upload is byte-exact."""
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.build(actor="researcher")
    assert builder.validate(expected_actor="researcher") == built_package.contract

    bundle = built_package.root / "bundle"
    with zipfile.ZipFile(built_package.root / "upload/bundle.zip") as archive:
        expected = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != builder.METADATA_NAME
        }
        assert set(archive.namelist()) == expected
        for name in expected:
            assert archive.getinfo(name).compress_type == zipfile.ZIP_STORED
            assert archive.read(name) == (bundle / name).read_bytes()


def test_rendered_kernel_is_two_files_compilable_and_below_kaggle_limit(
    built_package: BuiltPackage,
) -> None:
    """The executable envelope stays uploadable and has no undeclared sidecars."""
    kernel = built_package.root / "kernel"
    assert {path.name for path in kernel.iterdir()} == {
        "kernel-metadata.json",
        "run.py",
    }
    run_path = kernel / "run.py"
    assert run_path.stat().st_size < 1_000_000
    compile(run_path.read_bytes(), str(run_path), "exec")
    runtime = _module(run_path)
    assert (
        builder._sha256(
            built_package.root / "bundle" / builder.CONTRACT_NAME,
        )
        == runtime.INPUT_CONTRACT_SHA256
    )
    assert (
        built_package.contract["dataset_reference"] == runtime.INPUT_DATASET_REFERENCE
    )


def test_runtime_orchestration_keeps_branches_and_compiler_state_independent(
    built_package: BuiltPackage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each representation owns one process/GPU/cache and cannot gate its peer."""
    runtime = _module(built_package.root / "kernel/run.py")
    events: list[tuple[str, object, object]] = []

    class FakeProcess:
        def __init__(self, command: list[str], *, env: dict[str, str]) -> None:
            self.command = command
            self.env = env
            events.append(("spawn", command, env))

        def wait(self) -> int:
            events.append(("wait", self.command, self.env))
            return 0

    monkeypatch.setattr(runtime, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(runtime, "WORKING_ROOT", tmp_path / "working")
    monkeypatch.setattr(runtime.subprocess, "Popen", FakeProcess)
    returns = runtime._launch_children(
        tmp_path / "bundle",
        tmp_path / "bundle/mil_training_input.json",
        {"source/one": tmp_path / "source"},
        tmp_path / "bundle/initial_state.pt",
        {"contract": "hash"},
        None,
        17.0,
    )
    assert returns == {"normal_vae": 0, "so2_vae": 0}
    spawns = [event for event in events if event[0] == "spawn"]
    assert len(spawns) == 2
    assert [event[0] for event in events[:2]] == ["spawn", "spawn"]
    environments: dict[str, dict[str, str]] = {}
    for _, raw_command, raw_environment in spawns:
        command = cast("list[str]", raw_command)
        environment = cast("dict[str, str]", raw_environment)
        branch = command[command.index("--worker") + 1]
        environments[branch] = environment
    assert set(environments) == set(runtime.BRANCH_DEVICES)
    for branch, device in runtime.BRANCH_DEVICES.items():
        environment = environments[branch]
        assert environment["CUDA_VISIBLE_DEVICES"] == str(device)
        assert branch in environment["TORCHINDUCTOR_CACHE_DIR"]
        assert branch in environment["TRITON_CACHE_DIR"]
    assert (
        environments["normal_vae"]["TORCHINDUCTOR_CACHE_DIR"]
        != environments["so2_vae"]["TORCHINDUCTOR_CACHE_DIR"]
    )

    source = (built_package.root / "kernel/run.py").read_text(encoding="utf-8")
    assert runtime.BRANCH_DEVICES == {"normal_vae": 0, "so2_vae": 1}
    assert "DistributedDataParallel" not in source
    assert "torch.distributed" not in source
    assert "threading" not in source
    assert "barrier(" not in source


def test_resume_preflight_requires_best_and_completed_final_slots(
    built_package: BuiltPackage,
) -> None:
    """Resume must reject missing selector evidence before either paid branch starts."""
    runtime = _module(built_package.root / "kernel/run.py")
    metrics = SimpleNamespace(marker="best")
    active_selection = SimpleNamespace(
        best_boundary=53,
        best_metrics=metrics,
        stopped=False,
    )
    complete_selection = SimpleNamespace(
        best_boundary=53,
        best_metrics=metrics,
        stopped=False,
    )
    calls: list[str] = []

    def load_checkpoint(
        _root: Path,
        *,
        slot: str,
        expected_branch_name: str,
        expected_contract_hashes: dict[str, str],
    ) -> object:
        del expected_branch_name, expected_contract_hashes
        calls.append(slot)
        if slot == "best" and "active" in str(_root):
            message = "missing best"
            raise ValueError(message)
        selection = complete_selection if "complete" in str(_root) else active_selection
        committed = 53
        if "complete" in str(_root):
            committed = 15_900
        return SimpleNamespace(
            payload={
                "committed_update": committed if slot != "best" else 53,
                "best_selection": selection,
            },
        )

    def selection_from_payload(value: object) -> object:
        return value

    training = SimpleNamespace(
        TOTAL_UPDATES=15_900,
        load_branch_checkpoint=load_checkpoint,
        selection_from_checkpoint_payload=selection_from_payload,
    )
    with pytest.raises(ValueError, match="missing best"):
        runtime._prevalidate_resume(
            training,
            {"branch_roots": {"normal_vae": Path("active")}},
            {"contract": "hash"},
        )

    calls.clear()
    runtime._prevalidate_resume(
        training,
        {"branch_roots": {"normal_vae": Path("complete")}},
        {"contract": "hash"},
    )
    assert calls == ["latest", "best", "final"]


def test_best_revalidation_uses_classification_semantics_without_loss_tolerance(
    built_package: BuiltPackage,
) -> None:
    """Record finite CE drift while predictions and primary selection stay exact."""
    runtime = _module(built_package.root / "kernel/run.py")
    metrics = {
        "macro_f1": 0.8,
        "balanced_accuracy": 0.75,
        "accuracy": 0.75,
        "mean_ce": 0.5,
        "per_class": [],
        "confusion_matrix": [[1]],
        "wsi_count": 1,
    }
    stored = {
        "wsi_id": 10,
        "truth": 0,
        "prediction": 0,
        "cross_entropy": 0.5,
        "logits": [1.0, 0.0],
        "bag_size": 25,
        "graph_identity": "graph",
        "graph_degree": {"minimum": 1, "maximum": 4, "mean": 2.0},
    }
    observed = {**stored, "cross_entropy": 0.5001, "logits": [1.0001, 0.0]}
    history = [
        {"boundary": 53, "macro_f1": 0.8, "predictions": [stored]},
        {"boundary": 106, "macro_f1": 0.7, "predictions": []},
    ]
    observed_metrics = {**metrics, "mean_ce": 0.5001}
    comparison = runtime._best_revalidation_comparison(
        best_boundary=53,
        selection_metrics=metrics,
        validation_history=history,
        observed_rows=[observed],
        observed_metrics=observed_metrics,
    )
    assert comparison["status"] == "passed"
    assert comparison["mean_ce_delta"] == pytest.approx(0.0001)
    assert comparison["maximum_absolute_logit_delta"] == pytest.approx(0.0001)

    changed_prediction = {**observed, "prediction": 1}
    changed = runtime._best_revalidation_comparison(
        best_boundary=53,
        selection_metrics=metrics,
        validation_history=history,
        observed_rows=[changed_prediction],
        observed_metrics=observed_metrics,
    )
    assert changed["status"] == "failed"
    assert "wsi_truth_prediction_or_graph" in changed["mismatches"]

    tied = runtime._best_revalidation_comparison(
        best_boundary=53,
        selection_metrics=metrics,
        validation_history=[
            *history,
            {"boundary": 159, "macro_f1": 0.8, "predictions": []},
        ],
        observed_rows=[observed],
        observed_metrics=observed_metrics,
    )
    assert tied["status"] == "failed"
    assert "primary_metric_not_unique" in tied["mismatches"]

    nonfinite = runtime._best_revalidation_comparison(
        best_boundary=53,
        selection_metrics=metrics,
        validation_history=history,
        observed_rows=[
            {
                **observed,
                "cross_entropy": float("nan"),
                "logits": [float("inf"), 0.0],
            },
        ],
        observed_metrics={**observed_metrics, "mean_ce": float("nan")},
    )
    assert nonfinite["status"] == "failed"
    assert "nonfinite_output" in nonfinite["mismatches"]


def test_aggregate_rejects_best_prediction_evidence_not_bound_to_checkpoint(
    built_package: BuiltPackage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require paired bootstrap rows to be bound to each selected checkpoint."""
    runtime = _module(built_package.root / "kernel/run.py")
    monkeypatch.setattr(runtime, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "BOOTSTRAP_REPLICATES", 8)
    truths = [0] * 5 + [1] * 6 + [2] * 8 + [3] * 2 + [4] * 2
    for branch in runtime.BRANCH_DEVICES:
        rows = [
            {
                "wsi_id": 10_000 + index,
                "truth": truth,
                "prediction": truth,
                "cross_entropy": 0.25,
            }
            for index, truth in enumerate(truths)
        ]
        rows_sha256 = runtime._canonical_sha256(rows)
        identity = {
            "branch": branch,
            "best_boundary": 53,
            "best_checkpoint_sha256": f"{branch}-checkpoint",
            "best_manifest_sha256": f"{branch}-manifest",
            "contract_hashes": {"input": "hash"},
        }
        comparison = {
            "schema_version": "spec0036.best_revalidation_comparison.v1",
            "status": "passed",
            "mismatches": [],
            "stable_rows_match": True,
            "unique_wsi_rows_in_same_order": True,
            "discrete_metrics_match": True,
            "finite_outputs": True,
            "argmax_predictions_match": True,
            "selected_primary_macro_f1_is_unique": True,
            "prediction_changes": [],
            "observed_prediction_rows_sha256": rows_sha256,
        }
        branch_root = tmp_path / branch
        runtime._write_json(
            branch_root / "final_summary.json",
            {
                "status": "complete",
                "best_metrics": {"macro_f1": 1.0, "mean_ce": 0.3},
                "best_revalidation_metrics": {"macro_f1": 1.0, "mean_ce": 0.25},
                "best_revalidation_comparison": comparison,
                "best_prediction_rows_sha256": rows_sha256,
                **identity,
            },
        )
        runtime._write_json(
            branch_root / "best_validation_predictions.json",
            {
                "schema_version": "spec0036.best_validation_predictions.v1",
                **identity,
                "rows_sha256": rows_sha256,
                "rows": rows,
            },
        )
        runtime._write_json(
            branch_root / "best_revalidation.json",
            {
                "schema_version": "spec0036.best_revalidation.v1",
                **identity,
                "prediction_rows_sha256": rows_sha256,
                "selection_metrics": {"macro_f1": 1.0, "mean_ce": 0.3},
                "metrics": {"macro_f1": 1.0, "mean_ce": 0.25},
                "comparison": comparison,
            },
        )
    assert (
        runtime._aggregate(dict.fromkeys(runtime.BRANCH_DEVICES, 0))["status"]
        == "complete"
    )

    artifact_path = tmp_path / "so2_vae/best_validation_predictions.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutated = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutated["rows"][0]["prediction"] = 1
    runtime._write_json(artifact_path, mutated)
    with pytest.raises(RuntimeError, match="evidence binding"):
        runtime._aggregate(dict.fromkeys(runtime.BRANCH_DEVICES, 0))
    runtime._write_json(artifact_path, artifact)

    revalidation_path = tmp_path / "so2_vae/best_revalidation.json"
    revalidation = json.loads(revalidation_path.read_text(encoding="utf-8"))
    revalidation["comparison"]["finite_outputs"] = False
    runtime._write_json(revalidation_path, revalidation)
    with pytest.raises(RuntimeError, match="evidence binding"):
        runtime._aggregate(dict.fromkeys(runtime.BRANCH_DEVICES, 0))


def test_runtime_pins_upgrade_compile_and_complete_training_lifecycle(
    built_package: BuiltPackage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote entrypoint enforces the locked stack and complete-bag lifecycle."""
    runtime = _module(built_package.root / "kernel/run.py")
    source = (built_package.root / "kernel/run.py").read_text(encoding="utf-8")
    parent_source = inspect.getsource(runtime._run_parent)
    assert parent_source.index("install_pinned_torch()") < parent_source.index(
        "import torch",
    )
    assert runtime.PINNED_TORCH_VERSION == "2.14.0"
    assert runtime.PINNED_TORCH_CUDA == "13.0"
    install_calls: list[list[str]] = []

    def record_install(command: list[str]) -> None:
        install_calls.append(command)

    monkeypatch.setattr(runtime, "WORKING_ROOT", tmp_path)
    monkeypatch.setattr(runtime.subprocess, "check_call", record_install)
    runtime.install_pinned_torch()
    assert install_calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "torch==2.14.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ],
    ]

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"Tesla T4 {index}"

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (7, 5)

    fake_cuda = FakeCuda()
    fake_torch = SimpleNamespace(
        __version__="2.14.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=fake_cuda,
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9200)),
    )
    fingerprint = runtime._validate_runtime(
        fake_torch,
        driver_versions=["550.54.15", "550.54.15"],
    )
    assert fingerprint["devices"] == ["Tesla T4 0", "Tesla T4 1"]
    assert fingerprint["driver_versions"] == ["550.54.15", "550.54.15"]
    fake_torch.version.cuda = "12.8"
    with pytest.raises(RuntimeError, match="runtime differs"):
        runtime._validate_runtime(
            fake_torch,
            driver_versions=["550.54.15", "550.54.15"],
        )
    fake_torch.version.cuda = "13.0"

    for name in runtime.COMPILE_DISABLE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    runtime._reject_compile_disable_environment()
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    with pytest.raises(RuntimeError, match="compile-disabled"):
        runtime._reject_compile_disable_environment()
    monkeypatch.delenv("TORCH_COMPILE_DISABLE")

    compile_kwargs: dict[str, object] = {}

    class FakeCompileTorch:
        nn = SimpleNamespace(functional=SimpleNamespace(cross_entropy=None))

        @staticmethod
        def compile(
            function: object,
            *,
            backend: str,
            fullgraph: bool,
            dynamic: object,
            mode: str,
            recompile_limit: int,
            isolate_recompiles: bool,
        ) -> object:
            compile_kwargs.update({
                "backend": backend,
                "fullgraph": fullgraph,
                "dynamic": dynamic,
                "mode": mode,
                "recompile_limit": recompile_limit,
                "isolate_recompiles": isolate_recompiles,
            })
            return function

    numerical = runtime._make_numerical(FakeCompileTorch, object())
    assert callable(numerical)
    assert compile_kwargs == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": None,
        "mode": "max-autotune-no-cudagraphs",
        "recompile_limit": 3,
        "isolate_recompiles": True,
    }

    source = (builder.ROOT / builder.TEMPLATE_PATH).read_text(encoding="utf-8")
    assert "_assert_compiler_contract" not in source
    assert source.count('"compiler": _compiler_counts(torch)') >= 2
    assert "SESSION_LIMIT_SECONDS = 12 * 60 * 60" in source

    first_runtime = {"torch": "2.14.0+cu130", "driver_versions": ["550", "550"]}
    second_runtime = {"torch": "2.14.0+cu130", "driver_versions": ["551", "551"]}
    assert (
        runtime._contract_hashes(built_package.contract, first_runtime)[
            "runtime_sha256"
        ]
        != runtime._contract_hashes(built_package.contract, second_runtime)[
            "runtime_sha256"
        ]
    )

    training_config = cast(
        "dict[str, object]",
        built_package.contract["training_config"],
    )
    assert training_config["bootstrap_replicates"] == 10_000
    assert training_config["maximum_epochs"] == 150
    assert training_config["total_updates"] == 15_900
    assert training_config["warmup_epochs"] == 5
    assert training_config["warmup_updates"] == 530
    assert training_config["warmup_start_ratio"] == pytest.approx(0.1)
    assert training_config["warmup_start_lr"] == pytest.approx(2e-5)
    assert training_config["early_stopping_start_epoch"] == 50
    assert training_config["early_stopping_start_update"] == 5_300
    assert training_config["patience_epochs"] == 20
    assert training_config["patience_checks"] == 40
    optimizer = cast("dict[str, object]", training_config["optimizer"])
    assert optimizer["matrix_weight_decay"] == pytest.approx(5e-3)
    assert optimizer["other_weight_decay"] == pytest.approx(0.0)
    assert optimizer["no_decay_roles"] == [
        "bias",
        "normalization",
        "relative_attention_bias_or_offset",
        "global_cls_reg_tokens",
        "local_null_keys",
    ]
    required = (
        "maybe_mark_dynamic",
        "FullForegroundBagDataset",
        "expected_contract_sha256",
        'split="train"',
        'split="validation"',
        "save_branch_checkpoint",
        "best_revalidation",
        "diagnosis_stratified_paired_bootstrap",
    )
    assert all(marker in source for marker in required)


def test_finalization_records_evidence_and_preserves_stopped_checkpoint(
    built_package: BuiltPackage,
) -> None:
    """Evidence precedes verdict and terminal resumes do not rewrite checkpoints."""
    runtime = _module(built_package.root / "kernel/run.py")
    source = inspect.getsource(runtime._run_branch)
    finalization = source[source.index('if result["status"] != "paused":') :]
    assert finalization.index("_canonical_sha256(best_rows)") < finalization.index(
        "training.compute_validation_metrics",
    )
    assert finalization.index('branch_root / "best_revalidation.json"') < (
        finalization.index("changed classification semantics")
    )
    assert source.index("if already_terminal:") < source.index(
        "final_checkpoint = _write_branch_checkpoint",
    )
    assert 'terminal_reason = "early_stopping" if selection.stopped' in source


def test_runtime_preserves_structured_branch_numerical_error_details() -> None:
    """Terminal overflow evidence must survive the branch wrapper."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    error = BranchNumericalError(
        "overflow retries exhausted",
        details={"attempts": 3, "scale_backoffs": 3, "named_gradients": []},
    )
    try:
        raise error
    except BranchNumericalError as caught:
        record = runtime._exception_record(caught)
    assert record["error_type"] == "BranchNumericalError"
    assert record["error"] == "overflow retries exhausted"
    assert record["error_details"] == error.details
    assert "BranchNumericalError" in record["traceback"]


def test_runtime_keeps_compiler_cache_out_of_outputs_and_names_failures() -> None:
    """Compiler artifacts stay temporary; terminal evidence identifies the WSI."""
    source = (builder.ROOT / builder.TEMPLATE_PATH).read_text(encoding="utf-8")

    assert 'Path(tempfile.gettempdir()) / "spec0036_compile_cache" / branch' in source
    assert 'WORKING_ROOT / ".spec0036_compile_cache"' not in source
    for marker in (
        '"phase": "training"',
        '"committed_update_before_attempt": committed',
        '"within_epoch_cursor": cursor',
        '"wsi_id": bag.wsi_id',
        '"diagnosis": bag.diagnosis_label',
        '"bag_size": bag.instance_count',
        '"scaler_scale": float(scaler.get_scale())',
    ):
        assert marker in source


def test_local_resume_packaging_authenticates_resumable_and_carried_branches() -> None:
    """A continuation binds every resumable branch or its terminal peer evidence."""
    source = inspect.getsource(builder.build_resume)
    validation = inspect.getsource(builder.validate_resume)
    assert 'for branch in ("normal_vae", "so2_vae")' in source
    assert "load_branch_checkpoint" in source
    assert "expected_branch_name=branch" in source
    assert "expected_contract_hashes" in source
    assert "carried_branches" in source
    assert "at least one resumable branch" in source
    assert "carried_branches" in validation
    launcher = (builder.ROOT / "scripts/kaggle_kernel.sh").read_text()
    assert "build-mil-training-resume" in launcher
    assert "validate-mil-training-resume" in launcher
    assert "kaggle_api" not in inspect.getsource(builder.build_resume)
    forbidden = (
        'split="test"',
        "DistributedSampler",
        "DataLoader",
        "activation_checkpoint",
        "gradient_accumulation",
    )
    assert all(marker not in source for marker in forbidden)


def test_runtime_resume_requires_external_binding_and_confined_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mounted contract cannot self-authorize or escape its exact dataset root."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    input_root = tmp_path / "input"
    resume_root = input_root / "datasets" / "owner" / "resume-slug"
    branches = {
        "normal_vae": "branches/normal_vae/checkpoints",
        "so2_vae": "branches/so2_vae/checkpoints",
    }
    for relative in branches.values():
        checkpoint_root = resume_root / relative
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "latest").write_text("pointer", encoding="utf-8")
    files = {
        path.relative_to(resume_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": builder._sha256(path),
        }
        for path in resume_root.rglob("*")
        if path.is_file()
    }
    contract = {
        "schema_version": "spec0036.mil_training_resume.v1",
        "dataset_reference": "owner/resume-slug",
        "input_contract_sha256": "input-hash",
        "branches": branches,
        "carried_branches": {},
        "files": files,
    }
    contract_path = resume_root / builder.RESUME_CONTRACT_NAME
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(runtime, "INPUT_ROOT", input_root)
    monkeypatch.setattr(runtime, "RESUME_DATASET_REFERENCE", "owner/resume-slug")
    monkeypatch.setattr(runtime, "RESUME_CONTRACT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="externally bound"):
        runtime._resolve_optional_resume("input-hash")

    monkeypatch.setattr(
        runtime,
        "RESUME_CONTRACT_SHA256",
        builder._sha256(contract_path),
    )
    resolved = runtime._resolve_optional_resume("input-hash")
    assert resolved["path"] == contract_path.resolve()

    contract["branches"] = {**branches, "normal_vae": "../foreign/checkpoints"}
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "RESUME_CONTRACT_SHA256",
        builder._sha256(contract_path),
    )
    with pytest.raises(RuntimeError, match="escapes"):
        runtime._resolve_optional_resume("input-hash")


def test_runtime_rejects_unbound_or_ambiguous_resume_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh kernels and externally bound kernels reject foreign extra contracts."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    input_root = tmp_path / "input"
    for name in ("foreign-a", "foreign-b"):
        root = input_root / name
        root.mkdir(parents=True)
        (root / builder.RESUME_CONTRACT_NAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime, "INPUT_ROOT", input_root)
    monkeypatch.setattr(runtime, "RESUME_DATASET_REFERENCE", "")
    monkeypatch.setattr(runtime, "RESUME_CONTRACT_SHA256", "")
    with pytest.raises(RuntimeError, match="unbound"):
        runtime._resolve_optional_resume("input-hash")
    monkeypatch.setattr(runtime, "RESUME_DATASET_REFERENCE", "owner/resume")
    monkeypatch.setattr(runtime, "RESUME_CONTRACT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime._resolve_optional_resume("input-hash")


def test_runtime_resumes_only_available_branch_and_carries_failed_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One valid branch remains recoverable without rerunning its terminal peer."""
    runtime = _module(builder.ROOT / builder.TEMPLATE_PATH)
    output_root = tmp_path / "output"
    monkeypatch.setattr(runtime, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(runtime, "WORKING_ROOT", tmp_path)
    commands: list[list[str]] = []

    class Process:
        @staticmethod
        def wait() -> int:
            return 0

    def popen(command: list[str], *, env: dict[str, str]) -> Process:
        del env
        commands.append(command)
        return Process()

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    carried_path = tmp_path / "carried_summary.json"
    carried_path.write_text(
        json.dumps({"branch": "so2_vae", "status": "failed"}),
        encoding="utf-8",
    )
    resume = {
        "branch_roots": {"normal_vae": tmp_path / "normal-checkpoints"},
        "carried_summaries": {"so2_vae": carried_path},
    }
    return_codes = runtime._launch_children(
        tmp_path,
        tmp_path / "contract.json",
        {},
        tmp_path / "initial.pt",
        {},
        resume,
        1.0,
    )
    assert set(return_codes) == {"normal_vae"}
    assert len(commands) == 1
    assert commands[0][commands[0].index("--worker") + 1] == "normal_vae"

    normal_root = output_root / "normal_vae"
    normal_root.mkdir(parents=True)
    (normal_root / "final_summary.json").write_text(
        json.dumps({"branch": "normal_vae", "status": "paused"}),
        encoding="utf-8",
    )
    overall = runtime._aggregate(return_codes, resume)
    assert overall["status"] == "failed"
    assert overall["paired_bootstrap"] is None
    assert not (output_root / "paired_bootstrap.json").exists()
    assert overall["branches"]["so2_vae"]["status"] == "failed"


def test_resume_builder_requires_matching_launch_and_complete_output_receipts(
    tmp_path: Path,
) -> None:
    """Local checkpoint staging trusts only bytes bound to one versioned launch."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    payload_root = output_root / "spec0036_mil_training"
    payload_root.mkdir()
    input_reference = "source-owner/training-inputs"
    run_contract = {
        "schema_version": "spec0036.run_contract.v1",
        "input_dataset_reference": input_reference,
        "kernel_sources": list(builder.KERNEL_SOURCES),
        "contract_hashes": {"input_contract_sha256": "a" * 64},
    }
    (payload_root / "run_contract.json").write_text(
        json.dumps(run_contract),
        encoding="utf-8",
    )
    (payload_root / "normal_vae").mkdir()
    evidence = payload_root / "normal_vae/final_summary.json"
    evidence.write_text('{"status":"paused"}', encoding="utf-8")
    kernel_reference = "prior-owner/eqvae-local-global-mil-training/7"
    launch = {
        "schema_version": "eqvae.kaggle_kernel_launch.v1",
        "kernel_reference": kernel_reference,
        "kernel_id": "prior-owner/eqvae-local-global-mil-training",
        "actor": "prior-owner",
        "accepted_version": 7,
        "source_locators": {
            "dataset_sources": [input_reference, "prior-owner/prior-resume"],
            "kernel_sources": list(builder.KERNEL_SOURCES),
            "competition_sources": [],
            "model_sources": [],
        },
    }
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    files = {
        path.relative_to(output_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": builder._sha256(path),
        }
        for path in output_root.rglob("*")
        if path.is_file()
    }
    receipt = {
        "schema_version": "eqvae.kaggle_download.v1",
        "resource_kind": "kernel",
        "resource_reference": kernel_reference,
        "resource_owner": "prior-owner",
        "resource_slug": "eqvae-local-global-mil-training",
        "resource_version": 7,
        "files": files,
    }
    (output_root / "kaggle_output_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    _, _, observed_input, provenance, observed_files, observed_payload_root = (
        builder._validate_prior_output_authority(
            output_root=output_root,
            launch_receipt_path=launch_path,
        )
    )
    assert observed_input == input_reference
    assert provenance["kernel_reference"] == kernel_reference
    assert observed_files == files
    assert observed_payload_root == payload_root.resolve()

    evidence.write_text('{"status":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="bind every byte"):
        builder._validate_prior_output_authority(
            output_root=output_root,
            launch_receipt_path=launch_path,
        )
