"""Convert workspace-sheet millimetres to the frozen 3x3 cell names."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DEFAULT_SHEET_SPEC = (
    Path(__file__).resolve().parent.parent / "robot" / "assets" / "workspace_sheet.json"
)


@lru_cache(maxsize=None)
def _grid_geometry(spec_path: str) -> tuple[float, float, float]:
    grid = json.loads(Path(spec_path).read_text(encoding="utf-8"))["grid"]
    return float(grid["origin_mm"][0]), float(grid["origin_mm"][1]), float(grid["cell_mm"])


def mm_to_cell(
    x_mm: float,
    y_mm: float,
    spec_path: str | Path = DEFAULT_SHEET_SPEC,
) -> str | None:
    """Return the cell containing a sheet point, or ``None`` outside the grid.

    Decision D1 defines x from tag 0 toward tag 1 (A to C), and y from tag 0
    toward tag 3 (1 to 3). The outer ROI is half-open on its right and bottom
    edges, which assigns every internal boundary to the cell on that edge's
    right/bottom side without overlap.
    """
    x0, y0, cell_mm = _grid_geometry(str(Path(spec_path).resolve()))
    x = float(x_mm) - x0
    y = float(y_mm) - y0
    if not (0.0 <= x < 3.0 * cell_mm and 0.0 <= y < 3.0 * cell_mm):
        return None
    column = int(x // cell_mm)
    row = int(y // cell_mm)
    return f"{'ABC'[column]}{row + 1}"
