from __future__ import annotations

from typing import Callable
import logging
import os
import random
import sys

import numpy as np


def configure_cli_runtime() -> None:
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    random.seed(1337)
    np.random.seed(1337)


def execute_cli(main_fn: Callable[[], int | None], logger: logging.Logger | None = None) -> int:
    configure_cli_runtime()
    try:
        rc = main_fn()
        return int(rc) if rc is not None else 0
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - exercised in callers/tests
        if logger is not None:
            logger.error('Pipeline Failed: %s', exc, exc_info=True)
        raise
