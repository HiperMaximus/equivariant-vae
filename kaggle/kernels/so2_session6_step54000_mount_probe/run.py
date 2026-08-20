# ruff: noqa: CPY001, D100, INP001, S101, T201
import hashlib
import json
from pathlib import Path

checkpoint = Path(
    "/kaggle/input/datasets/maximshtefan/eqvae-so2-session6-step54000/step_054000.pt",
)
candidates = [str(path) for path in Path("/kaggle/input").rglob("step_054000.pt")]
assert checkpoint.is_file(), f"missing {checkpoint}; candidates={candidates}"
data = checkpoint.read_bytes()
report = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
assert report == {
    "bytes": 16440368,
    "sha256": "2ae4785571e2d1b4e690957e3cf74f749c7e273f1701ee274cc7b2b2e4a8742c",
}
Path("/kaggle/working/step54000_mount_probe.json").write_text(
    json.dumps(report) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": "pass", **report}))
