"""Convert workspace-sheet millimetres to the frozen 3x3 cell names.

This is the one place perception names cells, and it follows the printed sheet and the robot
(workspace_sheet.json "rows" A-C / "cols" 1-3, robot/perception/sheet.cell_center): the letter
is the ROW, along tag 0 -> tag 3 (A on the arm's edge, C on the student's edge), the number is
the COLUMN, along tag 0 -> tag 1 (1 to 3, left to right for a student opposite the arm). So
A1 sits at tag 0, A3 at tag 1, C3 at tag 2, C1 at tag 3.
"""

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


def grid_geometry(spec_path: str | Path = DEFAULT_SHEET_SPEC) -> tuple[float, float, float]:
    """(x0_mm, y0_mm, cell_mm): the grid's corner nearest tag 0 and its cell size."""
    return _grid_geometry(str(Path(spec_path).resolve()))


def cell_center_mm(cell: str, spec_path: str | Path = DEFAULT_SHEET_SPEC) -> tuple[float, float]:
    """Sheet mm at the centre of a cell, the inverse of mm_to_cell."""
    x0, y0, cell_mm = grid_geometry(spec_path)
    row, column = "ABC".index(cell[0]), int(cell[1]) - 1
    return x0 + (column + 0.5) * cell_mm, y0 + (row + 0.5) * cell_mm


def tag_corner_cells(spec_path: str | Path = DEFAULT_SHEET_SPEC) -> dict[int, str]:
    """Tag id -> the corner cell nearest it, from the sheet spec: the start-up self-check."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    corners = ("A1", "A3", "C1", "C3")
    out = {}
    for tag, t in spec["tags"].items():
        tx, ty = t["center_mm"]
        out[int(tag)] = min(corners, key=lambda c: (cell_center_mm(c, spec_path)[0] - tx) ** 2
                                                   + (cell_center_mm(c, spec_path)[1] - ty) ** 2)
    return out


def mm_to_cell(
    x_mm: float,
    y_mm: float,
    spec_path: str | Path = DEFAULT_SHEET_SPEC,
) -> str | None:
    """Return the cell containing a sheet point, or ``None`` outside the grid.

    The row letter follows y from tag 0 toward tag 3 (A to C), the column number follows
    x from tag 0 toward tag 1 (1 to 3). The outer ROI is half-open on its right and bottom
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
    return f"{'ABC'[row]}{column + 1}"


def edge_margin_mm(
    x_mm: float,
    y_mm: float,
    spec_path: str | Path = DEFAULT_SHEET_SPEC,
) -> float:
    """Distance from a sheet point to the nearest boundary of the cell containing it,
    or -1.0 outside the grid. A cube centred near a line may straddle two cells."""
    x0, y0, cell_mm = _grid_geometry(str(Path(spec_path).resolve()))
    x = float(x_mm) - x0
    y = float(y_mm) - y0
    if not (0.0 <= x < 3.0 * cell_mm and 0.0 <= y < 3.0 * cell_mm):
        return -1.0
    fx, fy = x % cell_mm, y % cell_mm
    return float(min(fx, cell_mm - fx, fy, cell_mm - fy))
