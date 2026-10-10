"""Regenerate the printable AprilTag grid sheet.

Source of truth for the physical layout: the same numbers perception/setup_gui.py
uses (TAG_CENTERS_IN, GRID_BBOX_IN, CELL_SIZE_IN, TAG_SIZE_IN), so the printed
sheet and the calibration tool can never drift apart.

This exists because the first hand-made version of the PDF had two problems:
  - grid lines drawn in light gray (0.6pt, 72% gray) that an overexposed camera
    can't see at all, making visual calibration checks impossible
  - cell labels using row-letter/column-number ("A1 A2 A3" across one row),
    the opposite of the README's column-letter/row-number convention ("A1" is
    always the front-left cell)

Run: python -m perception.generate_grid_pdf
"""

from __future__ import annotations

import pymupdf as fitz

from perception.setup_gui import (
    CELL_SIZE_IN,
    GRID_BBOX_IN,
    GRID_COLS,
    GRID_ROWS,
    TAG_CENTERS_IN,
    TAG_SIZE_IN,
)

SOURCE_PDF = "Apprentice_AprilTag_Grid.pdf"  # where the original tag images come from
OUT_PDF = "Apprentice_AprilTag_Grid.pdf"

PAGE_SIZE_IN = (8.5, 11.0)
TAG_IMAGE_XREFS = {0: 3, 1: 4, 2: 5, 3: 6}  # order matches TAG_CENTERS_IN keys, verified by bbox
GRID_LINE_WIDTH_PT = 2.5  # was 0.6pt; needs to be visible even when overexposed
GRID_LINE_COLOR = (0, 0, 0)  # was (0.72, 0.72, 0.72); needs to be visible even when overexposed


def in_to_pt(v: float) -> float:
    return v * 72.0


def main() -> None:
    src = fitz.open(SOURCE_PDF)
    tag_images = {tid: src.extract_image(xref) for tid, xref in TAG_IMAGE_XREFS.items()}
    src.close()  # release the file lock before writing back to the same path on Windows

    doc = fitz.open()
    page = doc.new_page(width=in_to_pt(PAGE_SIZE_IN[0]), height=in_to_pt(PAGE_SIZE_IN[1]))

    # Corner AprilTags, exactly where the original PDF had them.
    for tid, (cx, cy) in TAG_CENTERS_IN.items():
        half = TAG_SIZE_IN / 2
        rect = fitz.Rect(
            in_to_pt(cx - half), in_to_pt(cy - half), in_to_pt(cx + half), in_to_pt(cy + half)
        )
        page.insert_image(rect, stream=tag_images[tid]["image"])
        label_y = rect.y0 - 10 if cy < 5.5 else rect.y1 + 12
        page.insert_text((rect.x0, label_y), f"ID {tid}", fontsize=7, fontname="helv", color=(0, 0, 0))

    # Grid: thick black lines so the camera can actually see them, for a direct
    # visual check against perception.setup_gui's projected overlay.
    gx0, gy0, gx1, gy1 = (in_to_pt(v) for v in GRID_BBOX_IN)
    step = in_to_pt(CELL_SIZE_IN)
    shape = page.new_shape()
    for i in range(4):
        x = gx0 + i * step
        shape.draw_line((x, gy0), (x, gy1))
    for i in range(4):
        y = gy0 + i * step
        shape.draw_line((gx0, y), (gx1, y))
    shape.finish(color=GRID_LINE_COLOR, width=GRID_LINE_WIDTH_PT)
    shape.commit()

    # Cell labels: column letter (A-C, left to right) + row number (1-3, front
    # to back), matching README section 3 and perception/setup_gui.py's cell_at().
    for col_idx, col in enumerate(GRID_COLS):
        for row_idx, row in enumerate(GRID_ROWS):
            x = gx0 + col_idx * step + 7
            y = gy0 + row_idx * step + 12
            page.insert_text((x, y), f"{col}{row}", fontsize=7, fontname="helv", color=(0.55, 0.55, 0.55))

    # Ruler calibration mark and caption, copied from the original.
    ruler_y = gy1 + in_to_pt(1.93)
    shape = page.new_shape()
    shape.draw_line((gx0, ruler_y), (gx0 + in_to_pt(2.0), ruler_y))
    shape.draw_line((gx0, ruler_y - 4), (gx0, ruler_y + 4))
    shape.draw_line((gx0 + in_to_pt(2.0), ruler_y - 4), (gx0 + in_to_pt(2.0), ruler_y + 4))
    shape.finish(color=(0, 0, 0), width=1.0)
    shape.commit()
    page.insert_text(
        (gx0 + in_to_pt(2.15), ruler_y + 3),
        "Must measure exactly 2.00 in (50.8 mm). Print at 100% / Actual size, not Fit to page.",
        fontsize=7.5, fontname="helv", color=(0, 0, 0),
    )
    page.insert_text(
        (gx0, ruler_y + in_to_pt(0.15)),
        f"Cells {CELL_SIZE_IN:.2f} in ({CELL_SIZE_IN * 25.4:.1f} mm). Tags tag36h11 IDs 0-3, "
        f"black square {TAG_SIZE_IN:.2f} in. Matte paper, tape flat to a rigid board.",
        fontsize=7.5, fontname="helv", color=(0, 0, 0),
    )

    doc.save(OUT_PDF)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
