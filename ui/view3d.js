// Optional 3D board: the same state as the 2D board, with real stacked cubes, seen from the
// student's side of the grid. Loaded on demand by app.js; any failure keeps the 2D view.
import * as THREE from "./vendor/three.module.min.js";

const COLUMNS = ["A", "B", "C"];
const ROWS = ["1", "2", "3"];
const CELLS = COLUMNS.flatMap((c) => ROWS.map((r) => c + r));
const CUBE = { red: 0xe5484d, blue: 0x3b82f6, yellow: 0xf7c531, green: 0x22a75a };
const UNKNOWN = 0x6b7280;
const AMBER = 0xf5b041;
const SIZE = 0.84;   // cube edge relative to the cell pitch (48 mm cubes on 57 mm cells)

function cssColor(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v ? new THREE.Color(v) : new THREE.Color(fallback);
}

// Same convention as the 2D board: x left to right and z far to near, as the student sees it.
function cellXZ(cell, side) {
  const col = COLUMNS.indexOf(cell[0]), row = ROWS.indexOf(cell[1]);
  const [sx, sy] = side === "row1" ? [2 - col, 2 - row] : [col, row];
  return [sx - 1, sy - 1];
}

function label(text, color = "#f3f5f8", px = 96) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = color;
  g.font = `700 ${px}px Segoe UI, system-ui, sans-serif`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 64, 68);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthWrite: false }));
  sprite.scale.set(0.42, 0.42, 1);
  return sprite;
}

export function createView3D(container) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  if (!gl) throw new Error("WebGL is not available");

  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(cssColor("--bg", "#0d1015"));
  container.appendChild(canvas);
  for (const [text, where] of [["Robot side", "top"], ["Your side", "bottom"]]) {
    const tag = document.createElement("div");   // DOM labels: cubes never hide them
    tag.className = `edge-label ${where}`;
    tag.textContent = text;
    container.appendChild(tag);
  }
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.display = "block";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(0, 7.2, 7.4);    // above and behind the student's edge (near = +z)
  camera.lookAt(0, 0.7, 0.1);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-3, 7, 5);
  scene.add(sun);

  const base = new THREE.Mesh(new THREE.BoxGeometry(3.3, 0.08, 3.3), new THREE.MeshStandardMaterial({ color: 0x161b23 }));
  base.position.y = -0.05;
  scene.add(base);

  const tileMat = new THREE.MeshStandardMaterial({ color: cssColor("--panel-2", "#1e2530") });
  const tileGeo = new THREE.BoxGeometry(0.94, 0.02, 0.94);
  const cubeGeo = new THREE.BoxGeometry(SIZE, SIZE, SIZE);
  const edgeGeo = new THREE.EdgesGeometry(cubeGeo);
  const ringGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.0, 0.04, 1.0));

  const dynamic = new THREE.Group();
  scene.add(dynamic);
  const fixed = new THREE.Group();
  scene.add(fixed);

  let side = null;
  let ghost = null, fault = null;
  let active = false, raf = 0;

  function layoutFixed(newSide) {
    fixed.clear();
    for (const c of CELLS) {
      const [x, z] = cellXZ(c, newSide);
      const tile = new THREE.Mesh(tileGeo, tileMat);
      tile.position.set(x, 0, z);
      fixed.add(tile);
    }
    for (const col of COLUMNS) {
      const [x] = cellXZ(col + "1", newSide);
      const s = label(col);
      s.position.set(x, 0.05, 1.95);
      fixed.add(s);
    }
    for (const row of ROWS) {
      const [, z] = cellXZ("A" + row, newSide);
      const s = label(row);
      s.position.set(-1.95, 0.05, z);
      fixed.add(s);
    }
    side = newSide;
  }

  function clearDynamic() {
    dynamic.traverse((o) => { if (o.material) o.material.dispose(); });   // geometries are shared
    dynamic.clear();
  }

  function update(state, phase, newSide, faultVisible) {
    if (newSide !== side) layoutFixed(newSide);
    clearDynamic();
    ghost = fault = null;
    if (!state) return;
    for (const c of CELLS) {
      const [x, z] = cellXZ(c, side);
      state.board[c].forEach((color, level) => {
        const m = new THREE.Mesh(cubeGeo, new THREE.MeshStandardMaterial({ color: CUBE[color] ?? UNKNOWN, roughness: 0.6 }));
        m.position.set(x, 0.01 + SIZE / 2 + level * SIZE, z);
        m.add(new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 })));
        dynamic.add(m);
      });
    }
    if (phase !== "running") return;
    const e = state.expected;
    if (e && state.board[e.cell].length <= e.level) {
      const [x, z] = cellXZ(e.cell, side);
      ghost = new THREE.Mesh(cubeGeo, new THREE.MeshStandardMaterial({ color: CUBE[e.color] ?? UNKNOWN, transparent: true, opacity: 0.4, depthWrite: false }));
      ghost.position.set(x, 0.01 + SIZE / 2 + e.level * SIZE, z);
      ghost.add(new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0xffffff })));
      ghost.userData.strong = state.mode === "human" && state.hintLevel >= 2;
      dynamic.add(ghost);
    }
    if (faultVisible) {
      const [x, z] = cellXZ(state.faultCell, side);
      fault = new THREE.LineSegments(ringGeo, new THREE.LineBasicMaterial({ color: AMBER, transparent: true }));
      fault.position.set(x, 0.03, z);
      dynamic.add(fault);
    }
  }

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);

  function frame(t) {
    if (!active) return;
    raf = requestAnimationFrame(frame);
    const s = t / 1000;
    if (ghost) {
      const fast = ghost.userData.strong;
      const k = 0.5 + 0.5 * Math.sin(s * (fast ? 7 : 4));
      ghost.material.opacity = fast ? 0.25 + 0.6 * k : 0.3 + 0.35 * k;
      const sc = fast ? 0.95 + 0.08 * k : 1;
      ghost.scale.set(sc, sc, sc);
    }
    if (fault) fault.material.opacity = 0.45 + 0.55 * (0.5 + 0.5 * Math.sin(s * 4.5));
    renderer.render(scene, camera);
  }

  function setActive(on) {
    if (on === active) return;
    active = on;
    if (on) { resize(); raf = requestAnimationFrame(frame); } else cancelAnimationFrame(raf);
  }

  return { update, setActive };
}
