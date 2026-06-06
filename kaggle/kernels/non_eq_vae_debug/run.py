# Copyright (c) 2026 HiperMaximus
"""Placeholder Kaggle script kernel for the translatable VAE baseline."""

from __future__ import annotations

from typing import NoReturn

NOT_IMPLEMENTATION_READY = True
EXIT_MESSAGE = (
    "This Kaggle kernel scaffold is not implementation-ready. "
    "Write docs/behavior_inventory_kaggle.md, lock spec 0001, implement the "
    "real launcher, and remove NOT_IMPLEMENTATION_READY before pushing."
)


def main() -> NoReturn:
    """Exit until the Kaggle launcher is implementation-ready.

    Raises:
        SystemExit: Always, because the scaffold is not push-ready.

    """
    raise SystemExit(EXIT_MESSAGE)


if __name__ == "__main__":
    main()
