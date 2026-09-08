# Copyright 2026 HiperMaximus
"""Tests for the local Kaggle OAuth execution wrapper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import cast

_FAKE_TOKEN = "unit-test-secret-token"  # noqa: S105
_FAKE_USERNAME = "professor-account"
_EXPECTED_TEMP_FILE_MODE = oct(0o600)


def test_oauth_exec_uses_temp_token_file_without_token_in_argv_or_stdout(
    tmp_path: Path,
) -> None:
    """The helper hides the token from argv/stdout and removes the token file."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "report.json"

    completed = subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repo_root / "scripts" / "kaggle_oauth_exec.py"),
            "kernels",
            "status",
            "owner/kernel",
        ),
        cwd=repo_root,
        env=_test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert _FAKE_TOKEN not in completed.stdout
    assert _FAKE_TOKEN not in completed.stderr
    report = _load_report(report_path)
    assert report["argv"] == ["kernels", "status", "owner/kernel"]
    assert report["token_in_argv"] is False
    assert report["token_value"] == _FAKE_TOKEN
    assert report["token_mode"] == _EXPECTED_TEMP_FILE_MODE
    token_path = report["token_path"]
    assert isinstance(token_path, str)
    assert not Path(token_path).exists()


def test_oauth_identity_prints_only_saved_authenticated_username(
    tmp_path: Path,
) -> None:
    """OAuth actor discovery must not mint or expose an access token."""
    repo_root = Path(__file__).resolve().parents[1]
    sdk_root = _write_fake_kagglesdk(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sdk_root)
    env["FAKE_KAGGLE_TOKEN"] = _FAKE_TOKEN
    env["FAKE_KAGGLE_USERNAME"] = _FAKE_USERNAME

    completed = subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repo_root / "scripts" / "kaggle_oauth_exec.py"),
            "--print-oauth-username",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{_FAKE_USERNAME}\n"
    assert _FAKE_TOKEN not in completed.stdout
    assert _FAKE_TOKEN not in completed.stderr


def test_legacy_identity_reads_username_without_exposing_key(tmp_path: Path) -> None:
    """Legacy actor discovery reads only username from kaggle.json."""
    repo_root = Path(__file__).resolve().parents[1]
    sdk_root = _write_fake_kagglesdk(tmp_path)
    config_dir = tmp_path / "credentials"
    config_dir.mkdir()
    secret = "legacy-secret-key"  # noqa: S105
    (config_dir / "kaggle.json").write_text(
        json.dumps({"username": _FAKE_USERNAME, "key": secret}) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sdk_root)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    env.pop("KAGGLE_USERNAME", None)

    completed = subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(repo_root / "scripts" / "kaggle_oauth_exec.py"),
            "--print-legacy-username",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{_FAKE_USERNAME}\n"
    assert secret not in completed.stdout
    assert secret not in completed.stderr


def test_kaggle_kernel_status_uses_env_shebang_fresh_oauth_wrapper(
    tmp_path: Path,
) -> None:
    """The shell wrapper supports env-style Kaggle shebangs without raw fallback."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "report.json"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "status",
            "owner/kernel",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert _FAKE_TOKEN not in completed.stdout
    assert _FAKE_TOKEN not in completed.stderr
    report = _load_report(report_path)
    assert report["argv"] == ["kernels", "status", "owner/kernel"]
    assert report["token_value"] == _FAKE_TOKEN


def test_kaggle_kernel_identity_uses_the_logged_in_oauth_actor(tmp_path: Path) -> None:
    """The shell workflow exposes the actor, not a repository owner constant."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "unused-report.json"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "identity",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{_FAKE_USERNAME}\n"
    assert _FAKE_TOKEN not in completed.stdout
    assert _FAKE_TOKEN not in completed.stderr


def test_dataset_download_rejects_unversioned_reference_before_remote_call(
    tmp_path: Path,
) -> None:
    """Dataset provenance is validated before creating files or calling Kaggle."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "unexpected-report.json"
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"
    output_dir = tmp_path / "dataset-output"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "dataset-download",
            "external-owner/shared-data",
            str(output_dir),
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not report_path.exists()
    assert not output_dir.exists()


def test_output_launch_rejects_unversioned_receipt_before_remote_call(
    tmp_path: Path,
) -> None:
    """Kernel output cannot resolve mutable latest before receipt validation."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "unexpected-report.json"
    launch_receipt = tmp_path / "launch.json"
    launch_receipt.write_text(
        json.dumps(
            {
                "schema_version": "eqvae.kaggle_kernel_launch.v1",
                "actor": "professor-account",
                "kernel_id": "professor-account/job",
                "accepted_version": None,
                "kernel_reference": "professor-account/job",
            },
        ),
        encoding="utf-8",
    )
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"
    output_dir = tmp_path / "kernel-output"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "output-launch",
            str(launch_receipt),
            str(output_dir),
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not report_path.exists()
    assert not output_dir.exists()


def test_kaggle_kernel_api_check_does_not_use_raw_token_probe(
    tmp_path: Path,
) -> None:
    """The read-only preflight proves auth through wrapped endpoint calls."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_api_check_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "api-check-calls.jsonl"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "api-check",
            "kaggle/kernels/selected_runtime_debug",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "raw token probe invoked" not in completed.stderr
    assert _FAKE_TOKEN not in completed.stdout
    assert _FAKE_TOKEN not in completed.stderr
    calls = _load_reports(report_path)
    assert ["auth", "print-access-token"] not in calls
    assert [
        "kernels",
        "list",
        "--mine",
        "--search",
        "eqvae-selected-runtime-debug",
        "--csv",
    ] in calls
    portable_id = f"{_FAKE_USERNAME}/eqvae-selected-runtime-debug"
    assert ["kernels", "status", portable_id] in calls
    assert ["kernels", "logs", portable_id] in calls
    assert [
        "datasets",
        "files",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
        "-v",
    ] in calls
    assert ["quota", "-v"] in calls
    assert [
        "kernels",
        "files",
        portable_id,
        "-v",
    ] in calls


def test_api_check_accepts_any_first_launch_for_authenticated_actor(
    tmp_path: Path,
) -> None:
    """A professor's new kernel needs source checks but no status endpoint."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_api_check_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "first-launch-api-check.jsonl"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"
    env["FAKE_KAGGLE_MISSING_KERNEL"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "api-check",
            "kaggle/kernels/selected_runtime_debug",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "first launch may proceed" in completed.stdout
    calls = _load_reports(report_path)
    portable_id = f"{_FAKE_USERNAME}/eqvae-selected-runtime-debug"
    assert ["kernels", "status", portable_id] not in calls
    assert ["kernels", "logs", portable_id] not in calls
    assert ["kernels", "files", portable_id, "-v"] not in calls
    assert [
        "datasets",
        "files",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
        "-v",
    ] in calls


def test_atlas_api_check_accepts_first_launch_and_probes_exact_sources(
    tmp_path: Path,
) -> None:
    """A never-pushed atlas kernel has no status, but both raw mounts must exist."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_api_check_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "atlas-api-check-calls.jsonl"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"
    env["FAKE_KAGGLE_MISSING_KERNEL"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "api-check",
            "kaggle/kernels/ubc_ocean_test_atlas",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "first launch may proceed" in completed.stdout
    calls = _load_reports(report_path)
    assert [
        "kernels",
        "status",
        f"{_FAKE_USERNAME}/eqvae-ubc-ocean-test-atlas",
    ] not in calls
    assert [
        "kernels",
        "logs",
        f"{_FAKE_USERNAME}/eqvae-ubc-ocean-test-atlas",
    ] not in calls
    assert ["competitions", "files", "UBC-OCEAN", "-v"] in calls
    assert [
        "datasets",
        "files",
        "sohier/ubc-ovarian-cancer-competition-supplemental-masks",
        "-v",
    ] in calls


def test_atlas_resume_api_check_probes_private_checkpoint_source(
    tmp_path: Path,
) -> None:
    """A resume push must prove the optional private checkpoint is readable.

    Checking only the raw competition inputs would let an authorized push fail
    later during Kaggle source preparation, before the resumable script starts.
    """
    repo_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "repository"
    scripts_dir = isolated / "scripts"
    kernel_dir = isolated / "kaggle/kernels/ubc_ocean_test_atlas"
    scripts_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)
    shutil.copy2(repo_root / "scripts/kaggle_kernel.sh", scripts_dir)
    shutil.copy2(repo_root / "scripts/kaggle_oauth_exec.py", scripts_dir)
    metadata = cast(
        "dict[str, object]",
        json.loads(
            (
                repo_root / "kaggle/kernels/ubc_ocean_test_atlas/kernel-metadata.json"
            ).read_text(encoding="utf-8"),
        ),
    )
    metadata["dataset_sources"] = [
        "sohier/ubc-ovarian-cancer-competition-supplemental-masks",
        "maximusshtefan/eqvae-ubc-ocean-test-atlas-checkpoint",
    ]
    (kernel_dir / "kernel-metadata.json").write_text(
        f"{json.dumps(metadata)}\n",
        encoding="utf-8",
    )

    fake_bin = _write_fake_api_check_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "atlas-resume-api-check-calls.jsonl"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"
    env["FAKE_KAGGLE_MISSING_KERNEL"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(scripts_dir / "kaggle_kernel.sh"),
            "api-check",
            "kaggle/kernels/ubc_ocean_test_atlas",
        ),
        cwd=isolated,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = _load_reports(report_path)
    assert [
        "datasets",
        "files",
        "maximusshtefan/eqvae-ubc-ocean-test-atlas-checkpoint",
        "-v",
    ] in calls


def test_atlas_api_check_fails_when_listed_kernel_status_fails(tmp_path: Path) -> None:
    """An existing kernel's auth/network/status error must never look like absence."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = _write_fake_api_check_kaggle(tmp_path)
    sdk_root = _write_fake_kagglesdk(tmp_path)
    report_path = tmp_path / "atlas-status-error-calls.jsonl"
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = _test_env(fake_bin=fake_bin, sdk_root=sdk_root, report_path=report_path)
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"
    env["FAKE_KAGGLE_LIST_ATLAS"] = "1"
    env["FAKE_KAGGLE_STATUS_FAILURE"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "api-check",
            "kaggle/kernels/ubc_ocean_test_atlas",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "first launch may proceed" not in completed.stdout
    calls = _load_reports(report_path)
    assert [
        "kernels",
        "status",
        f"{_FAKE_USERNAME}/eqvae-ubc-ocean-test-atlas",
    ] in calls
    assert ["competitions", "files", "UBC-OCEAN", "-v"] not in calls


def test_kaggle_kernel_refuses_silent_raw_fallback_when_oauth_helper_unavailable(
    tmp_path: Path,
) -> None:
    """OAuth credentials plus an unresolvable Kaggle shebang fail clearly."""
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        "#!/definitely/missing/python\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text("{}\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["HOME"] = str(home)
    env["KAGGLE_REMOTE_CONFIRMED"] = "1"

    completed = subprocess.run(  # noqa: S603
        (
            _required_executable("bash"),
            str(repo_root / "scripts" / "kaggle_kernel.sh"),
            "status",
            "owner/kernel",
        ),
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "fresh-token wrapper" in completed.stderr


def _write_fake_kaggle(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(exist_ok=True)
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import stat\n"
        "import sys\n"
        "token_path = pathlib.Path(os.environ['KAGGLE_API_TOKEN'])\n"
        "token_value = token_path.read_text(encoding='utf-8')\n"
        "mode = stat.S_IMODE(token_path.stat().st_mode)\n"
        "report = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'token_in_argv': any(token_value in arg for arg in sys.argv),\n"
        "    'token_mode': oct(mode),\n"
        "    'token_path': str(token_path),\n"
        "    'token_value': token_value,\n"
        "}\n"
        "pathlib.Path(os.environ['FAKE_KAGGLE_REPORT']).write_text(\n"
        "    json.dumps(report, sort_keys=True),\n"
        "    encoding='utf-8',\n"
        ")\n"
        "sys.stdout.write('fake kaggle ok\\n')\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    return fake_bin


def _write_fake_api_check_kaggle(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(exist_ok=True)
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "argv = sys.argv[1:]\n"
        "if argv == ['--version']:\n"
        "    sys.stdout.write('Kaggle CLI test\\n')\n"
        "    raise SystemExit(0)\n"
        "if argv == ['auth', 'print-access-token']:\n"
        "    sys.stderr.write('raw token probe invoked\\n')\n"
        "    raise SystemExit(86)\n"
        "token_path = pathlib.Path(os.environ['KAGGLE_API_TOKEN'])\n"
        "token_value = token_path.read_text(encoding='utf-8')\n"
        "if token_value != os.environ['FAKE_KAGGLE_TOKEN']:\n"
        "    sys.stderr.write('wrong token value\\n')\n"
        "    raise SystemExit(87)\n"
        "report = pathlib.Path(os.environ['FAKE_KAGGLE_REPORT'])\n"
        "with report.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(argv) + '\\n')\n"
        "if (os.environ.get('FAKE_KAGGLE_LIST_ATLAS') == '1' and\n"
        "        argv[:2] == ['kernels', 'list']):\n"
        "    sys.stdout.write('ref,title\\n')\n"
        "    sys.stdout.write(os.environ['FAKE_KAGGLE_USERNAME'] + "
        "'/eqvae-ubc-ocean-test-atlas,atlas\\n')\n"
        "elif (os.environ.get('FAKE_KAGGLE_MISSING_KERNEL') != '1' and\n"
        "        argv[:2] == ['kernels', 'list']):\n"
        "    search = argv[argv.index('--search') + 1]\n"
        "    sys.stdout.write('ref,title\\n')\n"
        "    sys.stdout.write(os.environ['FAKE_KAGGLE_USERNAME'] + '/' + "
        "search + ',kernel\\n')\n"
        "if (os.environ.get('FAKE_KAGGLE_MISSING_KERNEL') == '1' and\n"
        "        argv[:2] == ['kernels', 'status']):\n"
        "    raise SystemExit(1)\n"
        "if (os.environ.get('FAKE_KAGGLE_STATUS_FAILURE') == '1' and\n"
        "        argv[:2] == ['kernels', 'status']):\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    return fake_bin


def _write_fake_kagglesdk(tmp_path: Path) -> Path:
    sdk_root = tmp_path / "fake_sdk"
    package = sdk_root / "kagglesdk"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import os\n"
        "class KaggleEnv:\n"
        "    PROD = 'PROD'\n"
        "class KaggleClient:\n"
        "    def __init__(self, env):\n"
        "        self.env = env\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return False\n"
        "class _Response:\n"
        "    def __init__(self, token):\n"
        "        self.token = token\n"
        "class KaggleCredentials:\n"
        "    @classmethod\n"
        "    def load(cls, client):\n"
        "        return cls()\n"
        "    def generate_access_token(self):\n"
        "        return _Response(os.environ['FAKE_KAGGLE_TOKEN'])\n"
        "    def get_username(self):\n"
        "        return os.environ['FAKE_KAGGLE_USERNAME']\n",
        encoding="utf-8",
    )
    return sdk_root


def _test_env(*, fake_bin: Path, sdk_root: Path, report_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{sdk_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(sdk_root)
    )
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_KAGGLE_TOKEN"] = _FAKE_TOKEN
    env["FAKE_KAGGLE_USERNAME"] = _FAKE_USERNAME
    env["FAKE_KAGGLE_REPORT"] = str(report_path)
    return env


def _load_report(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _load_reports(path: Path) -> list[list[str]]:
    return cast(
        "list[list[str]]",
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()],
    )


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    assert executable is not None
    return executable
