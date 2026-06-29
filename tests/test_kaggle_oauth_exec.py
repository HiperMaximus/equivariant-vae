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
        "        return _Response(os.environ['FAKE_KAGGLE_TOKEN'])\n",
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
    env["FAKE_KAGGLE_REPORT"] = str(report_path)
    return env


def _load_report(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    assert executable is not None
    return executable
