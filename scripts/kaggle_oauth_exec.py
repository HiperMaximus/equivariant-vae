# Copyright 2026 HiperMaximus
"""Run the Kaggle CLI with a freshly generated OAuth access token.

The Kaggle CLI can leave a cached OAuth access token in credentials.json that
still looks unexpired locally but is rejected by API endpoints. This helper uses
the stored refresh token to generate a fresh short-lived token and passes it to
the child CLI through a temporary token file, avoiding shell token expansion and
stdout token leaks.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess  # noqa: S404
import sys
import tempfile

from kagglesdk import KaggleClient, KaggleCredentials, KaggleEnv


def _fresh_oauth_token() -> str:
    with KaggleClient(env=KaggleEnv.PROD) as client:
        creds = KaggleCredentials.load(client=client)
        if creds is None:
            message = (
                "missing ~/.kaggle/credentials.json OAuth credentials; "
                "run `kaggle auth login`"
            )
            raise RuntimeError(message)
        response = creds.generate_access_token()
        if response is None or not response.token:
            message = (
                "unable to generate a Kaggle OAuth access token; "
                "run `kaggle auth login --force`"
            )
            raise RuntimeError(message)
        return response.token


def _write_error(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def main(argv: list[str]) -> int:
    """Run Kaggle CLI args with a freshly minted OAuth token.

    Returns:
        Child Kaggle CLI exit status.

    """
    if not argv:
        _write_error("usage: kaggle_oauth_exec.py <kaggle args...>")
        return 2

    token = _fresh_oauth_token()
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        _write_error("missing: kaggle")
        return 127

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="kaggle-api-token-",
    ) as token_file:
        pathlib.Path(token_file.name).chmod(stat.S_IRUSR | stat.S_IWUSR)
        token_file.write(token)
        token_file.flush()

        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = token_file.name
        return subprocess.run(  # noqa: S603
            [kaggle_bin, *argv],
            env=env,
            check=False,
        ).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        _write_error(f"error: {error}")
        raise SystemExit(2) from None
