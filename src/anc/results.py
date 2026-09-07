from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def prepare_output_dir(path: str | Path) -> Path:
    """
    Create and return an output directory.
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_array(
    output_dir: str | Path,
    filename: str,
    array: np.ndarray,
) -> Path:
    """
    Save a NumPy array as .npy.
    """
    output_dir = prepare_output_dir(output_dir)

    path = output_dir / filename
    np.save(path, np.asarray(array))

    return path


def save_json(
    output_dir: str | Path,
    filename: str,
    data: dict[str, Any],
) -> Path:
    """
    Save JSON-compatible experiment metadata/results.
    """
    output_dir = prepare_output_dir(output_dir)

    path = output_dir / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return path