# What we tried first

When an approach fails and we switch, add a row: idea, what happened, what we did instead.

| Idea | What happened | What we did instead |
|---|---|---|
| Standalone `perception/setup_gui.py`: interactive OpenCV tool to lock camera exposure, calibrate the grid homography, and sample cube colors | Camera turned out to have no software exposure control at all (confirmed by probing `.set()` then reading back the value on every property). Separately, `robot/` already had a working, more accurate camera pipeline (`robot/perception/`, 4.4mm mean placement error) that solved the same problem differently: it never tries to control exposure, it tunes HSV thresholds empirically to tolerate the overexposure. Its grid/tag geometry matched ours exactly even though measured independently. | Dropped `setup_gui.py`, `generate_grid_pdf.py`, and the regenerated grid PDF. Consolidating on `robot/perception/` as the one working camera pipeline instead of maintaining two. |
