# Copyright 2026 HiperMaximus
"""Transactional coordination for the two Spec 0021 latent writers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from eqvae.data.wsi_batches import WSIEvidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from eqvae.data.latent_shards import LatentRowIdentity, LatentShardWriter

type CatchUpModel = Literal["normal_vae", "so2_vae"]

WORKER_RESUME_SCHEMA: Final = "spec0021.dual_worker_resume.v1"
INCOMPLETE_SCHEMA: Final = "spec0021.dual_worker_incomplete.v1"
_SHA256_LENGTH: Final = 64


@dataclass(frozen=True)
class WorkerResumeBinding:
    """Immutable provenance bound into every worker resume update."""

    input_bundle_sha256: str
    run_config_sha256: str
    work_manifest_sha256: str
    run_number: int


@dataclass(frozen=True)
class ActiveWSIEvidence:
    """Journaled evidence written before either final writer commit."""

    row_start: int
    row_end: int
    evidence: WSIEvidence


@dataclass(frozen=True)
class CatchUpPlan:
    """The only accepted asymmetric restart: one lagging model, one WSI."""

    model_name: CatchUpModel
    row_start: int
    row_end: int
    evidence: WSIEvidence


@dataclass(frozen=True)
class VerificationPlan:
    """Require an independent WSI reread before journal promotion."""

    row_start: int
    row_end: int
    evidence: WSIEvidence


type RecoveryPlan = CatchUpPlan | VerificationPlan


@dataclass(frozen=True)
class _PairState:
    normal_rows: int
    normal_wsi_ids: tuple[int, ...]
    so2_rows: int
    so2_wsi_ids: tuple[int, ...]
    completed_evidence: tuple[WSIEvidence, ...]
    active_wsi: ActiveWSIEvidence | None


class DualLatentWriter:
    """Keep two shard writers lockstep with a durable active-WSI journal."""

    def __init__(
        self,
        *,
        normal_writer: LatentShardWriter,
        so2_writer: LatentShardWriter,
        state_path: Path,
        binding: WorkerResumeBinding,
    ) -> None:
        """Bind and validate the two exact shard writers and resume file.

        Raises:
            ValueError: If writers, binding, or an existing journal disagree.

        """
        self.normal_writer = normal_writer
        self.so2_writer = so2_writer
        self.state_path = state_path
        self.binding = binding
        self._catch_up: CatchUpPlan | None = None
        self._verification: VerificationPlan | None = None
        self._validate_static_contract()
        if state_path.exists():
            self._state = self._load_state()
        else:
            if normal_writer.committed_rows or so2_writer.committed_rows:
                message = "Committed latent prefix exists without worker resume state"
                raise ValueError(message)
            self._state = self._observed_state(active_wsi=None, completed=())
            self._write_state(self._state)

    @property
    def completed_wsi_evidence(self) -> tuple[WSIEvidence, ...]:
        """The durable ordered evidence for the converged WSI prefix."""
        return self._state.completed_evidence

    def reconcile(self) -> RecoveryPlan | None:
        """Validate startup prefixes and resolve any allowed asymmetry.

        Returns:
            A lagging-model or reread-verification plan, or None when ready.

        Raises:
            ValueError: If journal evidence and durable prefixes disagree.

        """
        self._validate_completed_evidence()
        normal_rows = self.normal_writer.committed_rows
        so2_rows = self.so2_writer.committed_rows
        active = self._state.active_wsi
        if normal_rows == so2_rows:
            if active is None:
                self._require_observed_prefixes_at_state()
            elif normal_rows == active.row_end:
                self._validate_active_journal(active)
                self._require_observed_prefixes_after_active(active)
                plan = VerificationPlan(
                    row_start=active.row_start,
                    row_end=active.row_end,
                    evidence=active.evidence,
                )
                self._verification = plan
                self._catch_up = None
                return plan
            elif normal_rows == active.row_start:
                self._validate_active_journal(active)
                self._require_observed_prefixes_at_state()
            else:
                message = "Equal writer prefixes disagree with active WSI journal"
                raise ValueError(message)
            self._catch_up = None
            self._verification = None
            return None

        if active is None:
            message = "Asymmetric writer prefixes have no active WSI evidence"
            raise ValueError(message)
        self._validate_active_journal(active)
        if normal_rows == active.row_end and so2_rows == active.row_start:
            model_name: CatchUpModel = "so2_vae"
            self._require_writer_prefix(
                self.normal_writer,
                rows=active.row_end,
                wsi_ids=(*self._state.normal_wsi_ids, active.evidence.wsi_id),
            )
            self._require_writer_prefix(
                self.so2_writer,
                rows=self._state.so2_rows,
                wsi_ids=self._state.so2_wsi_ids,
            )
        elif so2_rows == active.row_end and normal_rows == active.row_start:
            model_name = "normal_vae"
            self._require_writer_prefix(
                self.so2_writer,
                rows=active.row_end,
                wsi_ids=(*self._state.so2_wsi_ids, active.evidence.wsi_id),
            )
            self._require_writer_prefix(
                self.normal_writer,
                rows=self._state.normal_rows,
                wsi_ids=self._state.normal_wsi_ids,
            )
        else:
            message = "Writers differ by more than the one journaled complete WSI"
            raise ValueError(message)
        plan = CatchUpPlan(
            model_name=model_name,
            row_start=active.row_start,
            row_end=active.row_end,
            evidence=active.evidence,
        )
        self._catch_up = plan
        self._verification = None
        return plan

    def verify_converged_active(self, evidence: WSIEvidence) -> None:
        """Verify an independently reread WSI before promoting equal prefixes.

        Raises:
            RuntimeError: If reconciliation did not request verification.
            ValueError: If PNG or transcript evidence differs from the journal.

        """
        plan = self._verification
        if plan is None:
            message = "No converged active WSI requires verification"
            raise RuntimeError(message)
        if evidence != plan.evidence:
            message = "Converged active WSI reread evidence mismatch"
            raise ValueError(message)
        self._require_observed_prefixes_after_active(
            ActiveWSIEvidence(plan.row_start, plan.row_end, plan.evidence),
        )
        self._promote_active()
        self._verification = None

    def append_lockstep(
        self,
        *,
        row_start: int,
        identities: Sequence[LatentRowIdentity],
        normal_tensors: Tensor,
        so2_tensors: Tensor,
        final_wsi_evidence: WSIEvidence | None = None,
    ) -> None:
        """Submit one identical identity batch to both writers.

        Raises:
            RuntimeError: If catch-up is still pending.
            ValueError: If rows or final-WSI evidence are not lockstep.

        """
        if self._catch_up is not None or self._verification is not None:
            message = "Recovery plan must finish before lockstep appends"
            raise RuntimeError(message)
        if (
            self.normal_writer.next_row_start != row_start
            or self.so2_writer.next_row_start != row_start
        ):
            message = "Both writers must expect the same lockstep row_start"
            raise ValueError(message)
        row_end = row_start + len(identities)
        _wsi_id, wsi_start, wsi_end = self._range_for_row(row_start)
        ends_wsi = row_end == wsi_end
        if ends_wsi != (final_wsi_evidence is not None):
            message = "Final WSI batch and evidence must be supplied together"
            raise ValueError(message)
        if final_wsi_evidence is not None:
            self._prepare_active(
                ActiveWSIEvidence(
                    row_start=wsi_start,
                    row_end=wsi_end,
                    evidence=final_wsi_evidence,
                ),
            )
        self.normal_writer.append_batch(
            row_start=row_start,
            identities=identities,
            tensors=normal_tensors,
        )
        self.so2_writer.append_batch(
            row_start=row_start,
            identities=identities,
            tensors=so2_tensors,
        )
        if ends_wsi:
            self._promote_active()

    def append_catch_up(
        self,
        *,
        row_start: int,
        identities: Sequence[LatentRowIdentity],
        tensors: Tensor,
        final_wsi_evidence: WSIEvidence | None = None,
    ) -> None:
        """Append only to the lagging model named by ``reconcile``.

        Raises:
            RuntimeError: If no catch-up is pending or convergence fails.
            ValueError: If rows or reread evidence differ from the journal.

        """
        plan = self._catch_up
        if plan is None:
            message = "No lagging-model catch-up is active"
            raise RuntimeError(message)
        writer = (
            self.normal_writer if plan.model_name == "normal_vae" else self.so2_writer
        )
        if row_start != writer.next_row_start or row_start < plan.row_start:
            message = "Catch-up batch does not begin at the lagging prefix"
            raise ValueError(message)
        row_end = row_start + len(identities)
        if row_end > plan.row_end:
            message = "Catch-up batch overruns the one allowed WSI"
            raise ValueError(message)
        final = row_end == plan.row_end
        if final:
            if final_wsi_evidence != plan.evidence:
                message = "Catch-up PNG/transcript evidence mismatch"
                raise ValueError(message)
        elif final_wsi_evidence is not None:
            message = "Catch-up evidence may appear only on the final WSI batch"
            raise ValueError(message)
        writer.append_batch(
            row_start=row_start,
            identities=identities,
            tensors=tensors,
        )
        if final:
            if self.normal_writer.committed_rows != self.so2_writer.committed_rows:
                message = "Catch-up did not converge writer prefixes"
                raise RuntimeError(message)
            self._promote_active()
            self._catch_up = None

    def publish_incomplete(self, path: Path) -> None:
        """Rollback active tails and atomically publish recoverable prefixes."""
        self.normal_writer.rollback_uncommitted_wsi()
        self.so2_writer.rollback_uncommitted_wsi()
        plan = self.reconcile()
        payload: dict[str, object] = {
            "schema_version": INCOMPLETE_SCHEMA,
            "status": "incomplete",
            "worker_resume": self.state_path.name,
            "normal_completed_wsi_ids": list(
                self.normal_writer.completed_wsi_ids,
            ),
            "so2_completed_wsi_ids": list(self.so2_writer.completed_wsi_ids),
            "recovery_kind": (
                None
                if plan is None
                else "catch_up"
                if isinstance(plan, CatchUpPlan)
                else "verify_converged"
            ),
            "catch_up_model": (
                plan.model_name if isinstance(plan, CatchUpPlan) else None
            ),
        }
        _atomic_write_json(path, payload)

    def _prepare_active(self, active: ActiveWSIEvidence) -> None:
        if active.evidence.wsi_id != self._range_for_row(active.row_start)[0]:
            message = "Active WSI evidence names the wrong WSI"
            raise ValueError(message)
        if self._state.active_wsi is not None:
            if self._state.active_wsi != active:
                message = "Reread active WSI evidence differs from journal"
                raise ValueError(message)
            return
        self._state = _PairState(
            normal_rows=self._state.normal_rows,
            normal_wsi_ids=self._state.normal_wsi_ids,
            so2_rows=self._state.so2_rows,
            so2_wsi_ids=self._state.so2_wsi_ids,
            completed_evidence=self._state.completed_evidence,
            active_wsi=active,
        )
        self._write_state(self._state)

    def _promote_active(self) -> None:
        active = self._state.active_wsi
        if active is None:
            message = "No active WSI evidence is available to commit"
            raise RuntimeError(message)
        self._state = self._observed_state(
            active_wsi=None,
            completed=(*self._state.completed_evidence, active.evidence),
        )
        self._write_state(self._state)

    def _observed_state(
        self,
        *,
        active_wsi: ActiveWSIEvidence | None,
        completed: tuple[WSIEvidence, ...],
    ) -> _PairState:
        return _PairState(
            normal_rows=self.normal_writer.committed_rows,
            normal_wsi_ids=self.normal_writer.completed_wsi_ids,
            so2_rows=self.so2_writer.committed_rows,
            so2_wsi_ids=self.so2_writer.completed_wsi_ids,
            completed_evidence=completed,
            active_wsi=active_wsi,
        )

    def _validate_static_contract(self) -> None:
        if self.normal_writer.model_name != "normal_vae":
            message = "normal_writer must be bound to normal_vae"
            raise ValueError(message)
        if self.so2_writer.model_name != "so2_vae":
            message = "so2_writer must be bound to so2_vae"
            raise ValueError(message)
        normal_manifest = self.normal_writer.manifest
        so2_manifest = self.so2_writer.manifest
        if (
            normal_manifest.sha256 != so2_manifest.sha256
            or normal_manifest.run_number != so2_manifest.run_number
            or normal_manifest.rows != so2_manifest.rows
            or normal_manifest.wsi_ranges != so2_manifest.wsi_ranges
        ):
            message = "Dual writers are not bound to the same exact work manifest"
            raise ValueError(message)
        if (
            self.binding.work_manifest_sha256 != normal_manifest.sha256
            or self.binding.run_number != normal_manifest.run_number
        ):
            message = "Worker binding disagrees with writer work manifest"
            raise ValueError(message)
        for value in (
            self.binding.input_bundle_sha256,
            self.binding.run_config_sha256,
            self.binding.work_manifest_sha256,
        ):
            _validate_sha256(value)

    def _validate_completed_evidence(self) -> None:
        completed_ids = tuple(
            evidence.wsi_id for evidence in self._state.completed_evidence
        )
        if (
            completed_ids != self._state.normal_wsi_ids
            or completed_ids != self._state.so2_wsi_ids
            or self._state.normal_rows != self._state.so2_rows
        ):
            message = "Worker evidence does not match its converged WSI prefixes"
            raise ValueError(message)

    def _validate_active_journal(self, active: ActiveWSIEvidence) -> None:
        if (
            active.row_start != self._state.normal_rows
            or active.row_start != self._state.so2_rows
        ):
            message = "Active WSI does not begin at the converged worker prefix"
            raise ValueError(message)
        wsi_id, row_start, row_end = self._range_for_row(active.row_start)
        if (
            row_start != active.row_start
            or row_end != active.row_end
            or wsi_id != active.evidence.wsi_id
        ):
            message = "Active WSI journal disagrees with manifest boundaries"
            raise ValueError(message)

    def _require_observed_prefixes_at_state(self) -> None:
        self._require_writer_prefix(
            self.normal_writer,
            rows=self._state.normal_rows,
            wsi_ids=self._state.normal_wsi_ids,
        )
        self._require_writer_prefix(
            self.so2_writer,
            rows=self._state.so2_rows,
            wsi_ids=self._state.so2_wsi_ids,
        )

    def _require_observed_prefixes_after_active(
        self,
        active: ActiveWSIEvidence,
    ) -> None:
        self._require_writer_prefix(
            self.normal_writer,
            rows=active.row_end,
            wsi_ids=(*self._state.normal_wsi_ids, active.evidence.wsi_id),
        )
        self._require_writer_prefix(
            self.so2_writer,
            rows=active.row_end,
            wsi_ids=(*self._state.so2_wsi_ids, active.evidence.wsi_id),
        )

    @staticmethod
    def _require_writer_prefix(
        writer: LatentShardWriter,
        *,
        rows: int,
        wsi_ids: tuple[int, ...],
    ) -> None:
        if writer.committed_rows != rows or writer.completed_wsi_ids != wsi_ids:
            message = "Observed writer rows/WSI IDs disagree with worker resume state"
            raise ValueError(message)

    def _range_for_row(self, row_start: int) -> tuple[int, int, int]:
        for item in self.normal_writer.manifest.wsi_ranges:
            if item[1] <= row_start < item[2]:
                return item
        message = f"Row {row_start} lies outside the work manifest"
        raise IndexError(message)

    def _write_state(self, state: _PairState) -> None:
        payload: dict[str, object] = {
            "schema_version": WORKER_RESUME_SCHEMA,
            "status": "in_progress",
            "input_bundle_sha256": self.binding.input_bundle_sha256,
            "run_config_sha256": self.binding.run_config_sha256,
            "work_manifest_sha256": self.binding.work_manifest_sha256,
            "run_number": self.binding.run_number,
            "normal_prefix": _prefix_json(
                state.normal_rows,
                state.normal_wsi_ids,
            ),
            "so2_prefix": _prefix_json(state.so2_rows, state.so2_wsi_ids),
            "completed_wsi_evidence": [
                _evidence_json(evidence) for evidence in state.completed_evidence
            ],
            "active_wsi": (
                None if state.active_wsi is None else _active_json(state.active_wsi)
            ),
        }
        _atomic_write_json(self.state_path, payload)

    def _load_state(self) -> _PairState:
        raw = cast("object", json.loads(self.state_path.read_text(encoding="utf-8")))
        if not isinstance(raw, dict):
            message = "Worker resume state must be a JSON object"
            raise TypeError(message)
        state = cast("Mapping[str, object]", raw)
        expected_keys = {
            "schema_version",
            "status",
            "input_bundle_sha256",
            "run_config_sha256",
            "work_manifest_sha256",
            "run_number",
            "normal_prefix",
            "so2_prefix",
            "completed_wsi_evidence",
            "active_wsi",
        }
        if set(state) != expected_keys:
            message = "Worker resume fields do not match the canonical schema"
            raise ValueError(message)
        expected_static = {
            "schema_version": WORKER_RESUME_SCHEMA,
            "status": "in_progress",
            "input_bundle_sha256": self.binding.input_bundle_sha256,
            "run_config_sha256": self.binding.run_config_sha256,
            "work_manifest_sha256": self.binding.work_manifest_sha256,
            "run_number": self.binding.run_number,
        }
        for key, expected in expected_static.items():
            if state.get(key) != expected:
                message = f"Worker resume provenance mismatch for {key}"
                raise ValueError(message)
        normal_rows, normal_ids = _parse_prefix(state.get("normal_prefix"))
        so2_rows, so2_ids = _parse_prefix(state.get("so2_prefix"))
        evidence_raw = state.get("completed_wsi_evidence")
        if not isinstance(evidence_raw, list):
            message = "completed_wsi_evidence must be a list"
            raise TypeError(message)
        completed = tuple(
            _parse_evidence(item) for item in cast("list[object]", evidence_raw)
        )
        active_raw = state.get("active_wsi")
        active = None if active_raw is None else _parse_active(active_raw)
        return _PairState(
            normal_rows,
            normal_ids,
            so2_rows,
            so2_ids,
            completed,
            active,
        )


def _prefix_json(rows: int, wsi_ids: tuple[int, ...]) -> dict[str, object]:
    return {"committed_rows": rows, "completed_wsi_ids": list(wsi_ids)}


def _evidence_json(evidence: WSIEvidence) -> dict[str, object]:
    return {
        "wsi_id": evidence.wsi_id,
        "png_bytes": evidence.png_bytes,
        "png_sha256": evidence.png_sha256,
        "transcript_sha256": evidence.transcript_sha256,
    }


def _active_json(active: ActiveWSIEvidence) -> dict[str, object]:
    return {
        "row_start": active.row_start,
        "row_end": active.row_end,
        "evidence": _evidence_json(active.evidence),
    }


def _parse_prefix(value: object) -> tuple[int, tuple[int, ...]]:
    mapping = _mapping(value, "prefix")
    if set(mapping) != {"committed_rows", "completed_wsi_ids"}:
        message = "Worker prefix fields are invalid"
        raise ValueError(message)
    rows = _int(mapping, "committed_rows")
    raw_ids = mapping.get("completed_wsi_ids")
    if not isinstance(raw_ids, list):
        message = "completed_wsi_ids must be a list"
        raise TypeError(message)
    ids = tuple(
        _plain_int(item, "completed_wsi_id") for item in cast("list[object]", raw_ids)
    )
    return rows, ids


def _parse_evidence(value: object) -> WSIEvidence:
    mapping = _mapping(value, "WSI evidence")
    if set(mapping) != {
        "wsi_id",
        "png_bytes",
        "png_sha256",
        "transcript_sha256",
    }:
        message = "WSI evidence fields are invalid"
        raise ValueError(message)
    png_sha256 = _string(mapping, "png_sha256")
    transcript_sha256 = _string(mapping, "transcript_sha256")
    _validate_sha256(png_sha256)
    _validate_sha256(transcript_sha256)
    return WSIEvidence(
        wsi_id=_int(mapping, "wsi_id"),
        png_bytes=_int(mapping, "png_bytes"),
        png_sha256=png_sha256,
        transcript_sha256=transcript_sha256,
    )


def _parse_active(value: object) -> ActiveWSIEvidence:
    mapping = _mapping(value, "active_wsi")
    if set(mapping) != {"row_start", "row_end", "evidence"}:
        message = "active_wsi fields are invalid"
        raise ValueError(message)
    return ActiveWSIEvidence(
        row_start=_int(mapping, "row_start"),
        row_end=_int(mapping, "row_end"),
        evidence=_parse_evidence(mapping.get("evidence")),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        message = f"{name} must be an object"
        raise TypeError(message)
    return cast("Mapping[str, object]", value)


def _int(value: Mapping[str, object], key: str) -> int:
    return _plain_int(value.get(key), key)


def _plain_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"{key} must be a nonnegative integer"
        raise TypeError(message)
    return value


def _string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        message = f"{key} must be a string"
        raise TypeError(message)
    return result


def _validate_sha256(value: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in value
    ):
        message = f"Invalid SHA-256 value {value!r}"
        raise ValueError(message)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "INCOMPLETE_SCHEMA",
    "WORKER_RESUME_SCHEMA",
    "ActiveWSIEvidence",
    "CatchUpPlan",
    "DualLatentWriter",
    "RecoveryPlan",
    "VerificationPlan",
    "WorkerResumeBinding",
]
