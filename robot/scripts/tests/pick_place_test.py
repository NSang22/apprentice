"""Camera-guided pick and place of one cube.

    python -m robot.scripts.tests.pick_place_test red_cube B2             # red cube to cell B2
    python -m robot.scripts.tests.pick_place_test red_cube --onto blue_cube  # stack red on blue

Flow: hold the arm up -> it locks -> parks left of the sheet -> camera finds the cubes ->
you confirm the plan -> pick, place, park -> camera checks where the cube ended up.
"""

import argparse
import time

import cv2
import numpy as np

from robot.arm import Arm
from robot.correction import Correction
from robot.perception.camera import Camera
from robot.perception.config import ROOT
from robot.perception.scene import annotate, build_scene
from robot.perception.sheet import SheetLocalizer, cell_center
from robot.recorder import RunRecorder
from robot.session import ArmSession
from robot.skills import CUBE_MM, SkillError, Skills
from robot.workspace import Workspace

TOLERANCE_MM = 15.0


def look(cam, loc, tag):
    frame = cam.read()
    scene, pose, dets = build_scene(frame, loc)
    out = ROOT / "outputs" / f"pick_place_{tag}.png"
    cv2.imwrite(str(out), annotate(frame, scene, pose, dets))
    if scene is None:
        raise SkillError("camera can't see the sheet (need 2+ tags)")
    merged = [o["name"] for o in scene["objects"] if o["score"] < 0]
    if merged:
        print(f"  note: {merged} look like touching objects merged together; they won't be picked")
    return {o["name"]: o for o in scene["objects"] if o["score"] >= 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("cell", nargs="?")
    ap.add_argument("--onto")
    args = ap.parse_args()
    if not (args.cell or args.onto):
        ap.error("give a target cell (e.g. B2) or --onto <cube>")

    ws, loc = Workspace(), SheetLocalizer()
    cam = Camera()
    arm = Arm()
    rec = RunRecorder(cam, loc, ws, arm, label=f"{args.cube}_to_{args.onto or args.cell}")
    sk = Skills(arm, ws, log=lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}"), on_phase=rec,
                correct=Correction())
    t0 = time.time()
    try:
        with ArmSession(arm, sk):
            objs = look(cam, loc, "before")
            if args.cube not in objs:
                raise SkillError(f"camera doesn't see {args.cube}; sees {list(objs)}")
            src = objs[args.cube]
            if args.onto:
                if args.onto not in objs:
                    raise SkillError(f"camera doesn't see {args.onto}")
                dst_xy, dst_yaw, on = objs[args.onto]["center_mm"], objs[args.onto]["yaw_deg"], CUBE_MM
                target = f"on top of {args.onto}"
            else:
                dst_xy, dst_yaw, on = cell_center(args.cell), 0.0, 0.0
                target = f"cell {args.cell.upper()}"
            print(f"\nPlan: {args.cube} at ({src['center_mm'][0]:.0f}, {src['center_mm'][1]:.0f}) mm, "
                  f"yaw {src['yaw_deg']:.0f} -> {target} at ({dst_xy[0]:.0f}, {dst_xy[1]:.0f}) mm")
            rec("plan", cube_xy=src["center_mm"], target_xy=list(dst_xy))
            t_move = time.time()
            sk.pick(src["center_mm"], src["yaw_deg"])
            sk.place(dst_xy, dst_yaw, on)
            sk.park()
            print(f"  pick + place + park took {time.time() - t_move:.1f} s")

            rec("final", target_xy=list(dst_xy))
            objs = look(cam, loc, "after")
            if args.cube not in objs:
                print(f"\nRESULT: can't see {args.cube} after placing (fell off? hidden?). "
                      "See robot/outputs/pick_place_after.png")
            else:
                got = objs[args.cube]["center_mm"]
                err = float(np.hypot(got[0] - dst_xy[0], got[1] - dst_xy[1]))
                verdict = "PASS" if err <= TOLERANCE_MM else "FAIL"
                print(f"\nRESULT {verdict}: {args.cube} now at ({got[0]:.0f}, {got[1]:.0f}) mm, "
                      f"{err:.1f} mm from target (tolerance {TOLERANCE_MM:.0f}). See robot/outputs/pick_place_after.png")
    except SkillError as e:
        print(f"\nSTOPPED: {e}")
        rec("error", reason=str(e))
    finally:
        cam.close()
        print(f"Whole run {time.time() - t0:.0f} s. Saving evidence...")
        print(f"Run evidence: {rec.save()}")


if __name__ == "__main__":
    main()
