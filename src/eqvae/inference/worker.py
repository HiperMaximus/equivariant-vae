# Copyright 2026 HiperMaximus
# ruff: noqa: D102, D107, DOC201, DOC501, FBT003, PLR0914, PLR2004, PLW0717, TRY301
"""Executable dual-encoder worker for the Spec 0021 latent stores."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn

from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.data.latent_shards import (
    LATENT_CHANNELS,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    LatentArtifact,
    LatentRowIdentity,
    WorkManifest,
)
from eqvae.data.wsi_batches import WSIEvidence, WSIReader, iter_wsi_patch_batches
from eqvae.inference.dual_writer import (
    CatchUpPlan,
    DualLatentWriter,
    VerificationPlan,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

type NumericMode = Literal["FP32", "FP16-autocast"]
type ExecutionMode = Literal["eager", "compiled-fixed"]
type D2HMode = Literal["synchronous", "bounded-pinned-double-buffer"]

PAIR_AUDIT_SCHEMA: Final = "spec0021.latent_pair_audit.v1"


class _Encoder(Protocol):
    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class EncoderRecipe:
    """One selected encoder execution recipe."""

    batch_size: int
    numeric_mode: NumericMode
    execution: ExecutionMode
    d2h_mode: D2HMode = "synchronous"


@dataclass(frozen=True)
class EncoderTiming:
    """Synchronized stage timing for one encoded batch."""

    h2d_seconds: float
    normalization_seconds: float
    encoder_seconds: float
    d2h_seconds: float


@dataclass(frozen=True)
class WorkerResult:
    """Rows and WSIs durably completed during one invocation."""

    row_count: int
    wsi_count: int
    stopped_for_deadline: bool


@dataclass(frozen=True)
class DualEncoderTiming:
    """CUDA-event stage totals for one dual-device source batch."""

    h2d_seconds: float
    normalization_seconds: float
    normal_encoder_seconds: float
    so2_encoder_seconds: float
    d2h_seconds: float


@dataclass(frozen=True)
class DualEncodedBatch:
    """One occupied slot whose outputs remain owned until explicit release."""

    slot_index: int
    row_start: int
    identities: tuple[LatentRowIdentity, ...]
    normal_tensors: Tensor
    so2_tensors: Tensor
    final_wsi_evidence: WSIEvidence | None
    timing: DualEncoderTiming
    finite: bool
    identity_match: bool


@dataclass
class _DualSlot:
    input_buffer: Tensor
    normal_output: Tensor
    so2_output: Tensor
    occupied: bool = False
    row_start: int = 0
    identities: tuple[LatentRowIdentity, ...] = ()
    final_wsi_evidence: WSIEvidence | None = None
    valid_count: int = 0
    normal_events: tuple[torch.cuda.Event, ...] | None = None
    so2_events: tuple[torch.cuda.Event, ...] | None = None
    device_refs: list[Tensor] = field(default_factory=list)


class FrozenEncoderRunner:
    """Call only ``encode`` and return validated CPU FP32 posterior means."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
        recipe: EncoderRecipe,
    ) -> None:
        if recipe.batch_size < 1:
            message = "Encoder batch_size must be positive"
            raise ValueError(message)
        self.device = torch.device(device)
        self.recipe = recipe
        prepared = model.to(self.device).eval().requires_grad_(False)
        encode = cast("_Encoder", prepared).encode
        if recipe.execution == "compiled-fixed":
            encode = cast(
                "Callable[[Tensor], tuple[Tensor, Tensor]]",
                cast("object", torch.compile(encode)),  # pyright: ignore[reportUnknownMemberType]
            )
        self._encode = encode

    def __call__(self, images_uint8: Tensor) -> Tensor:
        """Encode one valid WSI-local batch, padding only for compiled shape."""
        output, _timing = self._run(images_uint8, measure=False)
        return output

    def encode_profiled(self, images_uint8: Tensor) -> tuple[Tensor, EncoderTiming]:
        """Encode one batch with synchronized per-stage timings."""
        return self._run(images_uint8, measure=True)

    def encode_normalized_device(self, normalized: Tensor) -> Tensor:
        """Run encoder-to-mu on an already normalized assigned-device tensor."""
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.recipe.numeric_mode == "FP16-autocast"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            mu, _logvar = self._encode(normalized)
        return mu

    def _run(
        self,
        images_uint8: Tensor,
        *,
        measure: bool,
    ) -> tuple[Tensor, EncoderTiming]:
        if images_uint8.ndim != 4 or images_uint8.shape[1:] != (3, 256, 256):
            message = "Encoder input must have shape (B,3,256,256)"
            raise ValueError(message)
        if images_uint8.dtype != torch.uint8:
            message = "Encoder input must be uint8"
            raise TypeError(message)
        valid_count = images_uint8.shape[0]
        if valid_count < 1 or valid_count > self.recipe.batch_size:
            message = "Encoder input count is outside the selected batch size"
            raise ValueError(message)
        source = images_uint8
        if (
            self.recipe.execution == "compiled-fixed"
            and valid_count < self.recipe.batch_size
        ):
            repeats = self.recipe.batch_size - valid_count
            source = torch.cat((source, source[-1:].expand(repeats, -1, -1, -1)))
        started = time.perf_counter()
        device_batch = source.to(self.device, non_blocking=True)
        self._synchronize()
        h2d_end = time.perf_counter()
        normalized = normalize_uint8_batch(device_batch)
        self._synchronize()
        normalization_end = time.perf_counter()
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.recipe.numeric_mode == "FP16-autocast"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            mu, _logvar = self._encode(normalized)
        self._synchronize()
        encoder_end = time.perf_counter()
        valid_mu = mu[:valid_count].detach().to(dtype=torch.float32)
        if (
            self.recipe.d2h_mode == "bounded-pinned-double-buffer"
            and self.device.type == "cuda"
        ):
            cpu_mu = torch.empty_like(valid_mu, device="cpu", pin_memory=True)
            cpu_mu.copy_(valid_mu, non_blocking=True)
        else:
            cpu_mu = valid_mu.to(device="cpu")
        self._synchronize()
        d2h_end = time.perf_counter()
        mu = cpu_mu
        expected = (valid_count, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
        if tuple(mu.shape) != expected:
            message = (
                f"Encoder posterior mu shape mismatch: {tuple(mu.shape)} != {expected}"
            )
            raise ValueError(message)
        if not torch.isfinite(mu).all():
            message = "Encoder posterior mu contains nonfinite values"
            raise ValueError(message)
        timing = EncoderTiming(
            h2d_seconds=h2d_end - started if measure else 0.0,
            normalization_seconds=(normalization_end - h2d_end) if measure else 0.0,
            encoder_seconds=(encoder_end - normalization_end) if measure else 0.0,
            d2h_seconds=(d2h_end - encoder_end) if measure else 0.0,
        )
        return mu.contiguous(), timing

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class DualDeviceExecutor:
    """Bounded shared-input executor for the two frozen encoders."""

    def __init__(
        self,
        *,
        normal: FrozenEncoderRunner,
        so2: FrozenEncoderRunner,
    ) -> None:
        if normal.recipe != so2.recipe:
            message = "Both encoders must use one identical execution recipe"
            raise ValueError(message)
        if (normal.device.type == "cuda") != (so2.device.type == "cuda"):
            message = "Both encoders must be either CUDA or CPU runners"
            raise ValueError(message)
        if normal.device == so2.device and normal.device.type == "cuda":
            message = "Dual CUDA encoders require distinct assigned devices"
            raise ValueError(message)
        self.normal = normal
        self.so2 = so2
        self.recipe = normal.recipe
        self.slot_count = (
            2 if self.recipe.d2h_mode == "bounded-pinned-double-buffer" else 1
        )
        self._cuda = normal.device.type == "cuda"
        self._normal_stream = (
            torch.cuda.Stream(device=normal.device) if self._cuda else None
        )
        self._so2_stream = torch.cuda.Stream(device=so2.device) if self._cuda else None
        self._slots = [self._make_slot() for _ in range(self.slot_count)]
        self._pending: list[int] = []

    @property
    def pending_count(self) -> int:
        """Submitted slots that have not yet been drained."""
        return len(self._pending)

    @property
    def pinned_bytes(self) -> int:
        """Exact capacity of all live executor-owned pinned buffers."""
        if not self._cuda:
            return 0
        return sum(
            slot.input_buffer.numel() * slot.input_buffer.element_size()
            + slot.normal_output.numel() * slot.normal_output.element_size()
            + slot.so2_output.numel() * slot.so2_output.element_size()
            for slot in self._slots
        )

    def submit(
        self,
        *,
        row_start: int,
        identities: tuple[LatentRowIdentity, ...],
        images_uint8: Tensor,
        final_wsi_evidence: WSIEvidence | None,
    ) -> None:
        """Fill one free shared-input slot and enqueue both device streams."""
        valid_count = len(identities)
        if (
            valid_count < 1
            or valid_count > self.recipe.batch_size
            or images_uint8.shape != (valid_count, 3, 256, 256)
            or images_uint8.dtype != torch.uint8
            or images_uint8.device.type != "cpu"
        ):
            message = "Dual executor input and identities violate the batch contract"
            raise ValueError(message)
        slot_index = next(
            (index for index, slot in enumerate(self._slots) if not slot.occupied),
            None,
        )
        if slot_index is None:
            message = "All bounded dual-encoder slots are still occupied"
            raise RuntimeError(message)
        slot = self._slots[slot_index]
        slot.input_buffer[:valid_count].copy_(images_uint8)
        effective_count = valid_count
        if (
            self.recipe.execution == "compiled-fixed"
            and valid_count < self.recipe.batch_size
        ):
            slot.input_buffer[valid_count:].copy_(slot.input_buffer[valid_count - 1])
            effective_count = self.recipe.batch_size
        slot.occupied = True
        slot.row_start = row_start
        slot.identities = identities
        slot.final_wsi_evidence = final_wsi_evidence
        slot.valid_count = valid_count
        slot.device_refs.clear()
        if self._cuda:
            normal_stream = cast("torch.cuda.Stream", self._normal_stream)
            so2_stream = cast("torch.cuda.Stream", self._so2_stream)
            slot.normal_events = self._enqueue_cuda(
                self.normal,
                normal_stream,
                slot.input_buffer[:effective_count],
                slot.normal_output,
                valid_count,
                slot.device_refs,
            )
            slot.so2_events = self._enqueue_cuda(
                self.so2,
                so2_stream,
                slot.input_buffer[:effective_count],
                slot.so2_output,
                valid_count,
                slot.device_refs,
            )
        else:
            slot.normal_output[:valid_count].copy_(self.normal(images_uint8))
            slot.so2_output[:valid_count].copy_(self.so2(images_uint8))
            slot.normal_events = None
            slot.so2_events = None
        self._pending.append(slot_index)

    def drain_one(self) -> DualEncodedBatch:
        """Synchronize and expose the oldest slot without releasing ownership."""
        if not self._pending:
            message = "No submitted dual-encoder slot is ready to drain"
            raise RuntimeError(message)
        slot_index = self._pending.pop(0)
        slot = self._slots[slot_index]
        timing = self._slot_timing(slot)
        count = slot.valid_count
        normal = slot.normal_output[:count]
        so2 = slot.so2_output[:count]
        expected_shape = (count, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
        identity_match = (
            len(slot.identities) == count
            and normal.shape == expected_shape
            and so2.shape == expected_shape
        )
        finite = bool(torch.isfinite(normal).all() and torch.isfinite(so2).all())
        if not identity_match:
            message = "Dual slot outputs no longer match their submitted identities"
            raise ValueError(message)
        if not finite:
            message = "Dual slot contains nonfinite posterior means"
            raise ValueError(message)
        return DualEncodedBatch(
            slot_index=slot_index,
            row_start=slot.row_start,
            identities=slot.identities,
            normal_tensors=normal,
            so2_tensors=so2,
            final_wsi_evidence=slot.final_wsi_evidence,
            timing=timing,
            finite=finite,
            identity_match=identity_match,
        )

    def release(self, batch: DualEncodedBatch) -> None:
        """Release one consumed slot, rejecting early or duplicate reuse."""
        slot = self._slots[batch.slot_index]
        if not slot.occupied or batch.slot_index in self._pending:
            message = "Dual slot may be released only after ordered drain"
            raise RuntimeError(message)
        slot.occupied = False
        slot.identities = ()
        slot.final_wsi_evidence = None
        slot.valid_count = 0
        slot.normal_events = None
        slot.so2_events = None
        slot.device_refs.clear()

    def abort(self) -> None:
        """Synchronize and release every outstanding slot after a failure."""
        if self._cuda:
            torch.cuda.synchronize(self.normal.device)
            torch.cuda.synchronize(self.so2.device)
        self._pending.clear()
        for slot in self._slots:
            slot.occupied = False
            slot.identities = ()
            slot.final_wsi_evidence = None
            slot.valid_count = 0
            slot.normal_events = None
            slot.so2_events = None
            slot.device_refs.clear()

    def _make_slot(self) -> _DualSlot:
        pin_memory = self._cuda
        batch_size = self.recipe.batch_size
        return _DualSlot(
            input_buffer=torch.empty(
                (batch_size, 3, 256, 256),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pin_memory,
            ),
            normal_output=torch.empty(
                (batch_size, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH),
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            ),
            so2_output=torch.empty(
                (batch_size, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH),
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            ),
        )

    @staticmethod
    def _events() -> tuple[torch.cuda.Event, ...]:
        return tuple(torch.cuda.Event(enable_timing=True) for _ in range(8))

    @classmethod
    def _enqueue_cuda(  # noqa: PLR0913, PLR0917
        cls,
        runner: FrozenEncoderRunner,
        stream: torch.cuda.Stream,
        source: Tensor,
        output: Tensor,
        valid_count: int,
        refs: list[Tensor],
    ) -> tuple[torch.cuda.Event, ...]:
        events = cls._events()
        with torch.cuda.device(runner.device), torch.cuda.stream(stream):
            events[0].record(stream)
            device_batch = source.to(runner.device, non_blocking=True)
            events[1].record(stream)
            normalized = normalize_uint8_batch(device_batch)
            events[2].record(stream)
            mu = runner.encode_normalized_device(normalized)
            events[3].record(stream)
            valid_mu = mu[:valid_count].detach().to(dtype=torch.float32)
            events[4].record(stream)
            output[:valid_count].copy_(valid_mu, non_blocking=True)
            events[5].record(stream)
            # Reserved event pairs keep the fixed timing schema extensible without
            # introducing host synchronization into the enqueued path.
            events[6].record(stream)
            events[7].record(stream)
        refs.extend((device_batch, normalized, mu, valid_mu))
        return events

    def _slot_timing(self, slot: _DualSlot) -> DualEncoderTiming:
        if not self._cuda:
            return DualEncoderTiming(0.0, 0.0, 0.0, 0.0, 0.0)
        normal = cast("tuple[torch.cuda.Event, ...]", slot.normal_events)
        so2 = cast("tuple[torch.cuda.Event, ...]", slot.so2_events)
        normal[5].synchronize()
        so2[5].synchronize()

        def elapsed(
            events: tuple[torch.cuda.Event, ...],
            start: int,
            end: int,
        ) -> float:
            return events[start].elapsed_time(events[end]) / 1000.0

        return DualEncoderTiming(
            h2d_seconds=elapsed(normal, 0, 1) + elapsed(so2, 0, 1),
            normalization_seconds=elapsed(normal, 1, 2) + elapsed(so2, 1, 2),
            normal_encoder_seconds=elapsed(normal, 2, 4),
            so2_encoder_seconds=elapsed(so2, 2, 4),
            d2h_seconds=elapsed(normal, 4, 5) + elapsed(so2, 4, 5),
        )


class DualEncoderWorker:
    """Drive two frozen encoders from one exact WSI batch stream."""

    def __init__(
        self,
        *,
        normal: FrozenEncoderRunner,
        so2: FrozenEncoderRunner,
    ) -> None:
        if normal.recipe.batch_size != so2.recipe.batch_size:
            message = "Both encoders must use the same source batch size"
            raise ValueError(message)
        self.normal = normal
        self.so2 = so2
        self.executor = DualDeviceExecutor(normal=normal, so2=so2)

    @property
    def batch_size(self) -> int:
        return self.normal.recipe.batch_size

    def run(  # noqa: PLR0913
        self,
        *,
        manifest: WorkManifest,
        wsi_dir: Path,
        writers: DualLatentWriter,
        incomplete_path: Path,
        reader: WSIReader | None = None,
        may_start_wsi: Callable[[int, int, int], bool] | None = None,
        expected_png_identities: Mapping[int, tuple[int, str]] | None = None,
    ) -> WorkerResult:
        """Recover if needed, then encode only at converged WSI boundaries."""
        completed_before = len(writers.normal_writer.completed_wsi_ids)
        self._recover(
            manifest=manifest,
            wsi_dir=wsi_dir,
            writers=writers,
            reader=reader,
            expected_png_identities=expected_png_identities,
        )
        start_row = writers.normal_writer.next_row_start
        stopped = False
        try:
            for wsi_id, wsi_start, wsi_end in manifest.wsi_ranges:
                if wsi_end <= start_row:
                    continue
                if wsi_start != writers.normal_writer.next_row_start:
                    message = "Worker may resume only at a converged WSI boundary"
                    raise ValueError(message)
                if may_start_wsi is not None and not may_start_wsi(
                    wsi_id,
                    wsi_start,
                    wsi_end,
                ):
                    stopped = True
                    break
                for batch in iter_wsi_patch_batches(
                    manifest=manifest,
                    wsi_dir=wsi_dir,
                    batch_size=self.batch_size,
                    start_row=wsi_start,
                    stop_row=wsi_end,
                    reader=reader,
                ):
                    self._require_expected_png(
                        batch.final_wsi_evidence,
                        expected_png_identities,
                    )
                    self.executor.submit(
                        row_start=batch.row_start,
                        identities=batch.identities,
                        images_uint8=batch.images_uint8,
                        final_wsi_evidence=batch.final_wsi_evidence,
                    )
                    if self.executor.pending_count == self.executor.slot_count:
                        self._drain_to_writer(writers)
                while self.executor.pending_count:
                    self._drain_to_writer(writers)
        except BaseException:
            self.executor.abort()
            writers.publish_incomplete(incomplete_path)
            raise
        if stopped:
            writers.publish_incomplete(incomplete_path)
        completed_after = len(writers.normal_writer.completed_wsi_ids)
        return WorkerResult(
            row_count=writers.normal_writer.committed_rows,
            wsi_count=completed_after - completed_before,
            stopped_for_deadline=stopped,
        )

    def _drain_to_writer(self, writers: DualLatentWriter) -> None:
        encoded = self.executor.drain_one()
        try:
            writers.append_lockstep(
                row_start=encoded.row_start,
                identities=encoded.identities,
                normal_tensors=encoded.normal_tensors,
                so2_tensors=encoded.so2_tensors,
                final_wsi_evidence=encoded.final_wsi_evidence,
            )
        finally:
            self.executor.release(encoded)

    def _recover(
        self,
        *,
        manifest: WorkManifest,
        wsi_dir: Path,
        writers: DualLatentWriter,
        reader: WSIReader | None,
        expected_png_identities: Mapping[int, tuple[int, str]] | None,
    ) -> None:
        plan = writers.reconcile()
        if plan is None:
            return
        last_evidence = None
        for batch in iter_wsi_patch_batches(
            manifest=manifest,
            wsi_dir=wsi_dir,
            batch_size=self.batch_size,
            start_row=plan.row_start,
            stop_row=plan.row_end,
            reader=reader,
        ):
            last_evidence = batch.final_wsi_evidence or last_evidence
            self._require_expected_png(
                batch.final_wsi_evidence,
                expected_png_identities,
            )
            if isinstance(plan, CatchUpPlan):
                runner = self.normal if plan.model_name == "normal_vae" else self.so2
                writers.append_catch_up(
                    row_start=batch.row_start,
                    identities=batch.identities,
                    tensors=runner(batch.images_uint8),
                    final_wsi_evidence=batch.final_wsi_evidence,
                )
        if isinstance(plan, VerificationPlan):
            if last_evidence is None:
                message = "Recovery reread produced no final WSI evidence"
                raise RuntimeError(message)
            writers.verify_converged_active(last_evidence)

    @staticmethod
    def _require_expected_png(
        evidence: WSIEvidence | None,
        expected: Mapping[int, tuple[int, str]] | None,
    ) -> None:
        if evidence is None or expected is None:
            return
        if expected.get(evidence.wsi_id) != (evidence.png_bytes, evidence.png_sha256):
            message = f"WSI {evidence.wsi_id} differs from the authoritative audit"
            raise ValueError(message)


def finalize_pair(
    *,
    writers: DualLatentWriter,
    pair_audit_path: Path,
    expected_union_sha256: str,
    run_config_sha256: str,
    input_receipt_sha256: str,
) -> dict[str, object]:
    """Finalize, fully validate, then atomically publish the pair audit last."""
    normal_writer = writers.normal_writer
    so2_writer = writers.so2_writer
    normal = normal_writer.finalize()
    so2 = so2_writer.finalize()
    if normal.manifest.rows != so2.manifest.rows:
        message = "Final latent pair manifests differ"
        raise ValueError(message)
    payload: dict[str, object] = {
        "schema_version": PAIR_AUDIT_SCHEMA,
        "status": "complete",
        "run_number": normal.manifest.run_number,
        "row_count": len(normal.manifest.rows),
        "work_manifest_sha256": normal.manifest.sha256,
        "union_manifest_sha256": expected_union_sha256,
        "run_config_sha256": run_config_sha256,
        "input_receipt_sha256": input_receipt_sha256,
        "completed_wsi_evidence": [
            asdict(evidence) for evidence in writers.completed_wsi_evidence
        ],
        "artifacts": {
            "normal_vae": _artifact_identity(normal),
            "so2_vae": _artifact_identity(so2),
        },
    }
    _atomic_publish_json(pair_audit_path, payload)
    return payload


def _artifact_identity(artifact: LatentArtifact) -> dict[str, object]:
    if artifact.file_sha256 is None:
        message = "Validated latent artifact lacks its full-file SHA-256"
        raise ValueError(message)
    return {
        "bin_name": artifact.bin_path.name,
        "bin_bytes": artifact.bin_path.stat().st_size,
        "bin_sha256": artifact.file_sha256,
        "sidecar_name": artifact.sidecar_path.name,
        "sidecar_bytes": artifact.sidecar_path.stat().st_size,
        "sidecar_sha256": _sha256_file(artifact.sidecar_path),
    }


def _atomic_publish_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()
    if path.exists():
        if path.read_bytes() != encoded:
            message = f"Existing audit differs: {path}"
            raise ValueError(message)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    Path(temporary).replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PAIR_AUDIT_SCHEMA",
    "DualDeviceExecutor",
    "DualEncodedBatch",
    "DualEncoderTiming",
    "DualEncoderWorker",
    "EncoderRecipe",
    "EncoderTiming",
    "FrozenEncoderRunner",
    "WorkerResult",
    "finalize_pair",
]
