"""Read complete historical Otsu WSI bags from the twelve paired FP16 shards."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECORD_SHAPE = (16, 32, 32)
RECORD_BYTES = 32768


@dataclass(frozen=True)
class BagLocation:
    shard: int
    first_record: int
    count: int
    diagnosis_index: int
    fold: int
    coordinates: tuple[tuple[int, int], ...]


class ShardBags:
    """A small WSI index; only the requested complete bag enters memory."""

    def __init__(
        self, root: Path, contract: dict, cohort_path: Path,
        expected_results: dict[int, str] | None = None,
    ) -> None:
        with cohort_path.open(newline="", encoding="utf-8") as handle:
            cohort = {int(row["wsi_id"]): row for row in csv.DictReader(handle)}
        self.locations: dict[int, BagLocation] = {}
        self.files: dict[str, dict[int, Path]] = {"normal_vae": {}, "so2_vae": {}}
        previous_atlas_row = -1
        previous_wsi = -1
        for shard in contract["shards"]:
            number = int(shard["shard"])
            suffix = f"shard_{number:02d}_of_12"
            matches = list(root.rglob(f"index_{suffix}.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one index_{suffix}.csv below {root}")
            directory = matches[0].parent
            result_path = directory / f"result_{suffix}.json"
            if expected_results is not None:
                with result_path.open("rb") as handle:
                    observed = hashlib.file_digest(handle, "sha256").hexdigest()
                if observed != expected_results[number]:
                    raise ValueError(f"Shard {number} result differs from the audited version")
            result = json.loads(result_path.read_text())
            index = matches[0]
            with index.open("rb") as handle:
                observed_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            if observed_hash != result["files"]["index"]["sha256"]:
                raise ValueError(f"Shard {number} index differs from its audited result")
            if result["rows_written"] != shard["rows"]:
                raise ValueError(f"Shard {number} row count differs")
            for branch in self.files:
                binary = directory / result["files"][branch]["name"]
                if binary.stat().st_size != result["files"][branch]["bytes"]:
                    raise ValueError(f"Shard {number} {branch} byte count differs")
                self.files[branch][number] = binary
            current_id = None
            current_first = 0
            coordinates: list[tuple[int, int]] = []
            current_label = 0
            current_fold = 0
            with index.open(newline="", encoding="utf-8") as handle:
                for file_index, row in enumerate(csv.DictReader(handle)):
                    wsi_id = int(row["wsi_id"])
                    atlas_row = int(row["atlas_row_index"])
                    if int(row["file_index"]) != file_index or atlas_row != previous_atlas_row + 1:
                        raise ValueError(f"Shard {number} row order differs")
                    previous_atlas_row = atlas_row
                    if current_id != wsi_id:
                        if current_id is not None:
                            self.locations[current_id] = BagLocation(
                                number, current_first, len(coordinates), current_label,
                                current_fold, tuple(coordinates),
                            )
                        if wsi_id <= previous_wsi:
                            raise ValueError("WSI order is not contiguous and increasing")
                        current_id = wsi_id
                        previous_wsi = wsi_id
                        current_first = file_index
                        current_label = int(row["diagnosis_index"])
                        current_fold = int(row["fold"])
                        coordinates = []
                    if int(row["diagnosis_index"]) != current_label or int(row["fold"]) != current_fold:
                        raise ValueError(f"WSI {wsi_id} metadata changes within its bag")
                    coordinate = (int(row["x"]), int(row["y"]))
                    if coordinates and (coordinate[1], coordinate[0]) <= (coordinates[-1][1], coordinates[-1][0]):
                        raise ValueError(f"WSI {wsi_id} coordinates are not y/x ordered")
                    coordinates.append(coordinate)
            if current_id is not None:
                self.locations[current_id] = BagLocation(
                    number, current_first, len(coordinates), current_label,
                    current_fold, tuple(coordinates),
                )
            if file_index + 1 != shard["rows"]:
                raise ValueError(f"Shard {number} index length differs")
        if set(self.locations) != set(cohort):
            raise ValueError("Shard WSI identities differ from the frozen cohort")
        for wsi_id, location in self.locations.items():
            row = cohort[wsi_id]
            if (location.diagnosis_index, location.fold) != (
                int(row["diagnosis_index"]), int(row["fold"])
            ):
                raise ValueError(f"WSI {wsi_id} differs from the frozen cohort")

    def read(self, branch: str, wsi_id: int) -> np.ndarray:
        location = self.locations[wsi_id]
        descriptor = os.open(self.files[branch][location.shard], os.O_RDONLY)
        try:
            payload = os.pread(
                descriptor, location.count * RECORD_BYTES,
                location.first_record * RECORD_BYTES,
            )
        finally:
            os.close(descriptor)
        if len(payload) != location.count * RECORD_BYTES:
            raise OSError(f"Incomplete FP16 bag read for WSI {wsi_id}")
        return np.frombuffer(payload, dtype="<f2").reshape((location.count, *RECORD_SHAPE))
