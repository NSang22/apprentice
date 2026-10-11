// Buddy's 3D board: the monitor's ui/view3d.js (served read-only at /buddy/ui/), fed from
// /api/buddy/state. mountBoard3D() resolves to null when WebGL or the module is unavailable, and
// Buddy keeps its 2D grid full size.
//
// Phase mapping:
//   show            the pattern as holographic cubes on its slots, the board otherwise empty
//   build, scan     the real board, live, no pattern (after a hint, the arrow stays on the hint cell)
//   helping         the real board; a hint: Pip's arm points at the hint cell under an arrow;
//                   a fix: the blocks coming off glow and the arm reaches for the first one
//   result          the real board, wrong blocks glowing like the monitor's fault cells, missing
//                   blocks as holograms
//   anything else   the real board

const CELLS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"];
const WRONG = ["wrong_color", "extra"];          // buddy/recall.py grade cells that glow

function boardOf(b) {
  const out = {};
  for (const c of CELLS) out[c] = Array.isArray(b && b[c]) ? b[c].filter((x) => typeof x === "string") : [];
  return out;
}

// What view3d's update() takes (see ui/app.js renderBoard), plus its optional Buddy fields.
export function viewFor(S) {
  const live = boardOf(S.board);
  const pattern = S.pattern || {};
  const v = { board: live, holograms: [], faults: [], pointAt: null, armAt: null };
  const ph = S.phase;
  if (ph === "show") {
    v.board = boardOf({});
    v.holograms = Object.entries(pattern).filter(([c, col]) => CELLS.includes(c) && col).map(([cell, color]) => ({ cell, level: 0, color }));
  } else if (ph === "helping") {
    if (S.hint && CELLS.includes(S.hint)) {
      v.pointAt = S.hint;
      v.armAt = { cell: S.hint, open: false };
    } else if ((S.removing || []).length) {
      v.faults = S.removing.filter((c) => CELLS.includes(c));
      if (v.faults.length) v.armAt = { cell: v.faults[0], open: true };
    }
  } else if (ph === "build" && S.hint && CELLS.includes(S.hint)) {
    v.pointAt = S.hint;               // the hint stays up while the child tries again, like the 2D glow
  } else if (ph === "result" && S.grade && S.grade.cells) {
    for (const [cell, res] of Object.entries(S.grade.cells)) {
      if (WRONG.includes(res)) v.faults.push(cell);
      else if (res === "missing" && pattern[cell]) v.holograms.push({ cell, level: 0, color: pattern[cell] });
    }
  }
  return v;
}

export async function mountBoard3D(container, colors) {
  let view;
  try {
    const mod = await import("/buddy/ui/view3d.js");
    view = mod.createView3D(container, { onDemand: true, maxPixelRatio: 2, lookX: 1.1, background: 0xeef6fb });
  } catch (err) {
    console.warn("[buddy] 3D board unavailable, keeping the 2D grid:", err);
    return null;
  }
  if (colors) view.setColors(colors);
  let key = null;
  return {
    update(S) {
      const v = viewFor(S);
      const k = JSON.stringify(v);
      if (k === key) return;          // the state polls every 300 ms: redraw only on a change
      key = k;
      const s = { board: v.board, expected: null, expectedCell: null, expectedLevel: null, finalBoard: null,
                  mode: "human", hintLevel: 0, faultCell: null };
      view.update({ s, phase: "running", side: "row3", faultVisible: false, arrow: null, lift: null, unsure: null,
                    activity: null, holograms: v.holograms, faults: v.faults, pointAt: v.pointAt, armAt: v.armAt });
    },
    setActive(on) { view.setActive(on); if (on) key = null; },
  };
}
