// 3D board, the monitor's main view: the real table as the student sees it. A wood tabletop,
// the white grid sheet with AprilTag corners, rounded foam cubes with soft shadows, a ghost
// cube with a light column, an arc arrow at hint level 2, amber floor glow on mistakes, and a
// low-poly arm on the robot side that reaches when the brain says the robot is busy.
// Purely visual: everything is driven by SessionState through update(). Loaded on demand by
// app.js; any failure (no WebGL) keeps the 2D board. All assets are drawn here at runtime.
import * as THREE from "./vendor/three.module.min.js";

// Cells follow the printed sheet: letter = row (A along the robot's edge), number = column.
const ROW_LETTERS = ["A", "B", "C"];
const COLUMN_NUMBERS = ["1", "2", "3"];
const CELLS = ROW_LETTERS.flatMap((r) => COLUMN_NUMBERS.map((c) => r + c));
const TAG_CORNERS = ["A1", "A3", "C3", "C1"];   // tag id -> the corner cell it sits at

// 1 unit = one cell (57.15 mm). Cubes are 48 mm foam.
const CUBE = 48 / 57.15;
const LIFT = 0.004;                       // cubes sit just above the paper (no z-fighting)
// Playskool silicone blocks; setColors() swaps in the measured colors from perception/calib/colors.json.
const CUBE_HEX = { red: 0xe2383f, orange: 0xf5862a, yellow: 0xf6c933, teal: 0x16a59b, skyblue: 0x62bdea, purple: 0x8a5cc7 };
const UNKNOWN_HEX = 0x7a808a;
const AMBER = 0xf5b041;
const OK_GREEN = 0x4ade80;
const PAPER = { w: 4.5, d: 5.3 };         // sheet size around the 3x3 grid
const ROBOT_BASE = new THREE.Vector3(0, 0, -3.35);  // robot side = far edge for the student

const DROP_MS = 380, FADE_MS = 320;
// Holograms (the per-step ghost and the intro preview) drop in from 1.5 cube heights with a small
// bounce. With prefers-reduced-motion they only fade in.
const HOLO_DROP_MS = 600, HOLO_FADE_MS = 500, GHOST_CYCLE_MS = 2000, HOLO_DROP_H = 1.5 * CUBE;
const reducedMotion = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

// Height above the slot, as a share of the drop: falls under gravity, then one small bounce.
function dropOffset(f) {
  if (f >= 1) return 0;
  if (f < 0.62) { const u = f / 0.62; return 1 - u * u; }
  return 0.1 * Math.sin(Math.PI * (f - 0.62) / 0.38);
}

// ---------------------------------------------------------------- geometry and textures

// Port of three.js examples RoundedBoxGeometry (MIT): a subdivided box whose vertices are
// pushed onto a rounded shell. Only positions and normals; the cubes are untextured.
function roundedBox(size, radius, segments = 3) {
  const segs = segments * 2 + 1;
  const g = new THREE.BoxGeometry(1, 1, 1, segs, segs, segs).toNonIndexed();
  const pos = g.attributes.position.array, nor = g.attributes.normal.array;
  const half = size / 2 - radius, step = 0.5 / segs;
  const p = new THREE.Vector3(), n = new THREE.Vector3();
  for (let i = 0; i < pos.length; i += 3) {
    p.fromArray(pos, i);
    n.set(p.x - Math.sign(p.x) * step, p.y - Math.sign(p.y) * step, p.z - Math.sign(p.z) * step).normalize();
    pos[i] = half * Math.sign(p.x) + n.x * radius;
    pos[i + 1] = half * Math.sign(p.y) + n.y * radius;
    pos[i + 2] = half * Math.sign(p.z) + n.z * radius;
    nor[i] = n.x; nor[i + 1] = n.y; nor[i + 2] = n.z;
  }
  g.attributes.position.needsUpdate = true;
  g.computeBoundingSphere();
  return g;
}

function canvasTexture(w, h, draw, srgb = true) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function woodTexture() {
  return canvasTexture(1024, 1024, (g, w, h) => {
    g.fillStyle = "#7a5638";
    g.fillRect(0, 0, w, h);
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < 260; i++) {            // long soft grain lines
      const y = rnd() * h, amp = 4 + rnd() * 14, freq = 0.002 + rnd() * 0.006, phase = rnd() * 6.28;
      g.strokeStyle = rnd() > 0.5 ? `rgba(40,24,12,${0.05 + rnd() * 0.08})` : `rgba(255,220,170,${0.03 + rnd() * 0.05})`;
      g.lineWidth = 1 + rnd() * 3;
      g.beginPath();
      for (let x = 0; x <= w; x += 16) g.lineTo(x, y + Math.sin(x * freq + phase) * amp);
      g.stroke();
    }
  });
}

function radialTexture(inner, outer) {
  return canvasTexture(256, 256, (g, w) => {
    const grd = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    grd.addColorStop(0, inner);
    grd.addColorStop(1, outer);
    g.fillStyle = grd;
    g.fillRect(0, 0, w, w);
  });
}

function columnTexture() {
  return canvasTexture(8, 256, (g, w, h) => {
    const grd = g.createLinearGradient(0, h, 0, 0);   // bright at the bottom, gone at the top
    grd.addColorStop(0, "rgba(255,255,255,0.85)");
    grd.addColorStop(0.35, "rgba(255,255,255,0.25)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd;
    g.fillRect(0, 0, w, h);
  });
}

function stripeTexture() {
  const t = canvasTexture(128, 8, (g, w, h) => {
    g.fillStyle = "#f5b041";
    g.fillRect(0, 0, w, h);
    g.fillStyle = "#fff1d6";
    g.fillRect(0, 0, w * 0.35, h);
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// Position of a cell as the student sees it: x left to right, z far (-) to near (+).
function cellXZ(cell, side) {
  const col = COLUMN_NUMBERS.indexOf(cell[1]), row = ROW_LETTERS.indexOf(cell[0]);
  const [sx, sy] = side === "row1" ? [2 - col, 2 - row] : [col, row];
  return [sx - 1, sy - 1];
}

// The sheet: white paper, thick black cell lines, AprilTag-style corners, labels upright for
// the student (canvas top = far edge, bottom = the student's edge).
function paperTexture(side) {
  const PX = 360;
  return canvasTexture(Math.round(PAPER.w * PX), Math.round(PAPER.d * PX), (g, w, h) => {
    const X = (u) => w / 2 + u * PX, Z = (v) => h / 2 + v * PX;
    g.fillStyle = "#f7f6f2";
    g.fillRect(0, 0, w, h);
    const grain = g.createLinearGradient(0, 0, w, h);   // faint paper shading
    grain.addColorStop(0, "rgba(0,0,0,0.02)");
    grain.addColorStop(1, "rgba(0,0,0,0.05)");
    g.fillStyle = grain;
    g.fillRect(0, 0, w, h);

    g.strokeStyle = "#111";
    g.lineWidth = 0.045 * PX;
    g.lineCap = "square";
    for (let i = 0; i <= 3; i++) {
      const t = -1.5 + i;
      g.beginPath(); g.moveTo(X(t), Z(-1.5)); g.lineTo(X(t), Z(1.5)); g.stroke();
      g.beginPath(); g.moveTo(X(-1.5), Z(t)); g.lineTo(X(1.5), Z(t)); g.stroke();
    }

    g.fillStyle = "#555";
    g.font = `600 ${0.24 * PX}px Segoe UI, system-ui, sans-serif`;
    g.textBaseline = "top";
    for (const c of CELLS) {
      const [x, z] = cellXZ(c, side);
      g.fillText(c, X(x - 0.42), Z(z - 0.42));
    }
    g.fillStyle = "#222";
    g.font = `700 ${0.4 * PX}px Segoe UI, system-ui, sans-serif`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    for (const col of COLUMN_NUMBERS) g.fillText(col, X(cellXZ("A" + col, side)[0]), Z(1.85));
    for (const row of ROW_LETTERS) g.fillText(row, X(-1.85), Z(cellXZ(row + "1", side)[1]));
    g.fillStyle = "#8a8a8a";
    g.font = `700 ${0.24 * PX}px Segoe UI, system-ui, sans-serif`;
    g.fillText("YOUR SIDE", X(0), Z(2.32));

    // Four corner tags: black square, white ring, a fixed 4x4 bit pattern per id.
    const patterns = [0b1011010011100101, 0b0110100111010011, 0b1100101101100110, 0b0011110010101101];
    // Each tag in the paper corner next to its cell (tag 0 at A1, 1 at A3, 2 at C3, 3 at C1).
    const corners = TAG_CORNERS.map((c) => cellXZ(c, side)).map(([x, z]) => [Math.sign(x) * 1.82, Math.sign(z) * 2.2]);
    corners.forEach(([u, v], id) => {
      const s = 0.62 * PX, x0 = X(u) - s / 2, y0 = Z(v) - s / 2, cellPx = s / 8;
      g.fillStyle = "#111";
      g.fillRect(x0, y0, s, s);
      g.fillStyle = "#f7f6f2";
      for (let k = 0; k < 16; k++) {
        if (patterns[id] & (1 << k)) g.fillRect(x0 + (2 + (k % 4)) * cellPx, y0 + (2 + Math.floor(k / 4)) * cellPx, cellPx, cellPx);
      }
    });
  });
}

// ---------------------------------------------------------------- the arm

function makeArm() {
  const white = new THREE.MeshStandardMaterial({ color: 0xe9e6df, roughness: 0.7, flatShading: true });
  const dark = new THREE.MeshStandardMaterial({ color: 0x2a2d33, roughness: 0.6, flatShading: true });
  const shadowy = (m) => { m.castShadow = true; m.receiveShadow = true; return m; };
  const L1 = 2.05, L2 = 1.9, SHOULDER_Y = 0.55;

  const root = new THREE.Group();
  root.position.copy(ROBOT_BASE);
  root.add(shadowy(new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.5, 0.22, 10), dark)));
  const yaw = new THREE.Group();
  yaw.position.y = 0.11;
  root.add(yaw);
  const turret = shadowy(new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.36, 0.44, 8), white));
  turret.position.y = 0.22;
  yaw.add(turret);

  const shoulder = new THREE.Group();
  shoulder.position.y = SHOULDER_Y;
  yaw.add(shoulder);
  shoulder.add(shadowy(new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.3, 0.3), dark)));
  const link1 = shadowy(new THREE.Mesh(new THREE.BoxGeometry(0.22, L1, 0.26), white));
  link1.position.y = L1 / 2;
  shoulder.add(link1);

  const elbow = new THREE.Group();
  elbow.position.y = L1;
  shoulder.add(elbow);
  elbow.add(shadowy(new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.26, 0.26), dark)));
  const link2 = shadowy(new THREE.Mesh(new THREE.BoxGeometry(0.18, L2, 0.22), white));
  link2.position.y = L2 / 2;
  elbow.add(link2);

  const wrist = new THREE.Group();
  wrist.position.y = L2;
  elbow.add(wrist);
  const palm = shadowy(new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.16, 0.2), dark));
  palm.position.y = 0.08;
  wrist.add(palm);
  const fingers = [-1, 1].map((sgn) => {
    const f = shadowy(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.34, 0.16), white));
    f.position.set(sgn * 0.13, 0.31, 0);
    wrist.add(f);
    return f;
  });

  const cur = { yaw: 0, a1: 0, a2: 0, a3: 0, open: 1 };
  const goal = { ...cur };

  // Point the gripper straight down at a world point (2-link planar IK, elbow up).
  function solve(target) {
    const dx = target.x - ROBOT_BASE.x, dz = target.z - ROBOT_BASE.z;
    const d = Math.hypot(dx, dz), h = target.y - (ROBOT_BASE.y + 0.11 + SHOULDER_Y);
    const r = Math.min(Math.hypot(d, h), (L1 + L2) * 0.985);
    const c2 = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    const a2 = Math.acos(Math.max(-1, Math.min(1, c2)));
    const phi = Math.atan2(d, h);
    const a1 = phi - Math.atan2(L2 * Math.sin(a2), L1 + L2 * Math.cos(a2));
    return { yaw: Math.atan2(dx, dz), a1, a2, a3: Math.PI - a1 - a2 };
  }

  // Resting: reaching a little forward and to one side, so the bent arm reads from the student's view.
  const REST = new THREE.Vector3(ROBOT_BASE.x - 0.9, 2.2, ROBOT_BASE.z + 1.6);
  Object.assign(cur, solve(REST));
  Object.assign(goal, cur);

  function aim(target, open) {
    Object.assign(goal, solve(target || REST));
    goal.open = open;
  }

  function step(dt) {
    const k = 1 - Math.exp(-dt * 3.2);   // smooth, never snappy
    for (const key of ["yaw", "a1", "a2", "a3", "open"]) cur[key] += (goal[key] - cur[key]) * k;
    yaw.rotation.y = cur.yaw;
    shoulder.rotation.x = cur.a1;
    elbow.rotation.x = cur.a2;
    wrist.rotation.x = cur.a3;
    fingers[0].position.x = -0.07 - 0.07 * cur.open;
    fingers[1].position.x = 0.07 + 0.07 * cur.open;
  }

  return { root, aim, step };
}

// ---------------------------------------------------------------- view

export function createView3D(container) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  if (!gl) throw new Error("WebGL is not available");

  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  canvas.style.cssText = "width:100%;height:100%;display:block";
  container.appendChild(canvas);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1015);
  scene.fog = new THREE.Fog(0x0d1015, 14, 30);

  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 80);
  // Aim left of the board so it sits right of the instruction overlay.
  const look = new THREE.Vector3(-1.5, 0.1, -0.35);

  // Lights: soft key with shadows, a cool fill, low ambient.
  scene.add(new THREE.HemisphereLight(0xfff6ea, 0x1b2230, 0.55));
  const key = new THREE.DirectionalLight(0xfff1de, 2.2);
  key.position.set(-4.5, 9, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -5; key.shadow.camera.right = 5;
  key.shadow.camera.top = 5; key.shadow.camera.bottom = -5;
  key.shadow.camera.near = 2; key.shadow.camera.far = 22;
  key.shadow.bias = -0.0006;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 4;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xcfe0ff, 0.55);
  fill.position.set(5, 4, 6);
  scene.add(fill);

  // Table and sheet.
  const wood = woodTexture();
  wood.wrapS = wood.wrapT = THREE.RepeatWrapping;
  wood.repeat.set(2, 2);
  const table = new THREE.Mesh(new THREE.BoxGeometry(16, 0.4, 11), new THREE.MeshStandardMaterial({ map: wood, roughness: 0.8 }));
  table.position.set(0, -0.2, -0.8);
  table.receiveShadow = true;
  scene.add(table);

  const board = new THREE.Group();        // turns slowly on the ready and complete screens
  scene.add(board);
  const paperMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
  const paper = new THREE.Mesh(new THREE.PlaneGeometry(PAPER.w, PAPER.d), paperMat);
  paper.rotation.x = -Math.PI / 2;
  paper.position.y = 0.001;
  paper.receiveShadow = true;
  board.add(paper);

  const cubeGeo = roundedBox(CUBE, 0.07, 3);
  const edgeGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(CUBE * 1.04, CUBE * 1.04, CUBE * 1.04));
  const aoTex = radialTexture("rgba(0,0,0,0.55)", "rgba(0,0,0,0)");
  const aoGeo = new THREE.PlaneGeometry(CUBE * 1.55, CUBE * 1.55);
  const glowTex = radialTexture("rgba(245,176,65,0.9)", "rgba(245,176,65,0)");
  const colTex = columnTexture();
  const stripes = stripeTexture();

  const arm = makeArm();
  scene.add(arm.root);

  // ------------------------------------------------------------ dynamic content

  const cubes = new Map();     // "A1:0" -> { mesh, color, born, dying }
  const intro = { active: false, holos: [], fading: 0 };   // the intro preview's holograms
  let side = null, ghost = null, column = null, arrow = null, lift = null, glows = [];
  let phase = "nodata", spin = 0, spinGoal = 0, pulses = [], burst = null;
  let active = false, raf = 0, last = 0;

  function cellPos(cell, level) {
    const [x, z] = cellXZ(cell, side);
    return new THREE.Vector3(x, LIFT + CUBE / 2 + level * CUBE, z);
  }

  const unknownTex = canvasTexture(128, 128, (g, w, h) => {
    g.fillStyle = "rgba(20,24,30,0.85)";
    g.beginPath(); g.arc(w / 2, h / 2, w / 2 - 4, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#fff";
    g.font = "800 92px Segoe UI, system-ui, sans-serif";
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText("?", w / 2, h / 2 + 6);
  });

  function makeCube(color) {
    const mat = new THREE.MeshStandardMaterial({ color: CUBE_HEX[color] ?? UNKNOWN_HEX, roughness: 0.92, metalness: 0, transparent: true });
    const mesh = new THREE.Mesh(cubeGeo, mat);
    if (!(color in CUBE_HEX)) {   // unknown: gray block with a "?" that always faces the camera
      const mark = new THREE.Sprite(new THREE.SpriteMaterial({ map: unknownTex, depthTest: false }));
      mark.scale.set(0.55, 0.55, 1);
      mark.position.y = CUBE * 0.95;
      mark.renderOrder = 11;
      mesh.add(mark);
    }
    mesh.castShadow = mesh.receiveShadow = true;
    const ao = new THREE.Mesh(aoGeo, new THREE.MeshBasicMaterial({ map: aoTex, transparent: true, depthWrite: false }));
    ao.rotation.x = -Math.PI / 2;
    ao.position.y = -CUBE / 2 + 0.002;
    ao.renderOrder = 1;
    mesh.add(ao);
    return mesh;
  }

  function setCubes(target) {
    const now = performance.now();
    const want = new Map();
    for (const c of CELLS) (target[c] || []).forEach((color, level) => want.set(`${c}:${level}`, color));
    // Cubes that no longer match fade out under a unique key, freeing their slot.
    for (const [k, cube] of [...cubes]) {
      if (cube.dying || want.get(k) === cube.color) continue;
      cube.dying = now;
      cube.mesh.material.transparent = true;
      cube.mesh.material.needsUpdate = true;
      cubes.delete(k);
      cubes.set(`${k}#${now}`, cube);
    }
    for (const [k, color] of want) {
      if (cubes.has(k)) continue;
      const [cell, level] = k.split(":");
      const mesh = makeCube(color);
      mesh.position.copy(cellPos(cell, Number(level)));
      board.add(mesh);
      cubes.set(k, { mesh, color, born: now, dying: 0, rest: mesh.position.y });
    }
  }

  function disposeObj(o) {
    o.traverse((n) => {
      if (n.material) (Array.isArray(n.material) ? n.material : [n.material]).forEach((m) => m.dispose());
      if (n.geometry && n.geometry !== cubeGeo && n.geometry !== edgeGeo && n.geometry !== aoGeo) n.geometry.dispose();
    });
    o.parent && o.parent.remove(o);
  }

  function clearHints() {
    for (const o of [ghost, column, arrow, lift, ...glows]) if (o) disposeObj(o);
    ghost = column = arrow = lift = null;
    glows = [];
  }

  // EXTRA: an arrow rising straight up off the block ("take it off"), never toward another cell.
  function makeLift(pos) {
    const group = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.95, depthTest: false });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.7, 16), mat);
    shaft.position.y = 0.35;
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.24, 0.42, 20), mat.clone());
    head.position.y = 0.9;
    for (const m of [shaft, head]) { m.renderOrder = 10; group.add(m); }
    group.position.set(pos.x, pos.y + CUBE * 0.6, pos.z);
    group.userData.base = group.position.y;
    board.add(group);
    return group;
  }

  // A translucent holographic cube with glowing edges: the ghost and the intro preview.
  function makeHologram(color) {
    const g = new THREE.Mesh(cubeGeo, new THREE.MeshStandardMaterial({
      color: CUBE_HEX[color] ?? UNKNOWN_HEX, transparent: true, opacity: 0.35, depthWrite: false,
      emissive: CUBE_HEX[color] ?? UNKNOWN_HEX, emissiveIntensity: 0.35, roughness: 0.6,
    }));
    g.add(new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })));
    return g;
  }

  function makeGhost(color, pos, strong) {
    const g = makeHologram(color);
    g.position.copy(pos);
    g.userData = { base: pos.y, strong, born: performance.now() };
    board.add(g);
    const col = new THREE.Mesh(new THREE.CylinderGeometry(CUBE * 0.5, CUBE * 0.62, 3.2, 24, 1, true),
      new THREE.MeshBasicMaterial({ map: colTex, color: CUBE_HEX[color] ?? 0xffffff, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    col.position.set(pos.x, pos.y - CUBE / 2 + 1.6, pos.z);
    board.add(col);
    return [g, col];
  }

  function makeArrow(from, to) {
    const a = from.clone(), b = to.clone();
    a.y += CUBE * 0.6; b.y += CUBE * 0.95;            // land just above the target slot
    const mid = a.clone().lerp(b, 0.5);
    mid.y += 1.5 + 0.3 * a.distanceTo(b);
    // Seen end on (from and to in line with the camera), a vertical arc collapses into a pole.
    // Swing the arc sideways in proportion, so it always reads as a curve with a visible head.
    const dir = new THREE.Vector3(b.x - a.x, 0, b.z - a.z).normalize();
    const view = new THREE.Vector3(camera.position.x - mid.x, 0, camera.position.z - mid.z).normalize();
    const along = Math.abs(dir.dot(view));
    if (along > 0.5) {
      const side = new THREE.Vector3(-dir.z, 0, dir.x);
      if (side.dot(new THREE.Vector3(1, 0, 0)) < 0) side.negate();   // swing toward open table on the right
      mid.addScaledVector(side, 1.6 * along);
      mid.y -= 0.5 * along;
    }
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    const group = new THREE.Group();
    const tex = stripes.clone();
    tex.needsUpdate = true;
    tex.repeat.set(Math.max(3, Math.round(curve.getLength() * 3)), 1);
    // Drawn over everything: the arrow must never hide behind a cube or the ghost.
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.075, 10, false),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.95, depthTest: false }));
    tube.renderOrder = 10;
    group.add(tube);
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.62, 20), new THREE.MeshBasicMaterial({ color: AMBER, depthTest: false }));
    head.renderOrder = 10;
    const tangent = curve.getTangent(1).normalize();
    head.position.copy(b);
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
    group.add(head);
    group.userData.tex = tex;
    board.add(group);
    return group;
  }

  function makeGlow(cell) {
    const [x, z] = cellXZ(cell, side);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(2.0, 2.0),
      new THREE.MeshBasicMaterial({ map: glowTex, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, 0.006, z);
    board.add(m);
    return m;
  }

  // view: { s, phase, side, faultVisible, arrow: [from, to] | null, lift: cell | null, unsure: cell | null, activity }
  function update(view) {
    const s = view.s;
    if (view.side !== side) {
      side = view.side;
      if (paperMat.map) paperMat.map.dispose();
      paperMat.map = paperTexture(side);
      paperMat.needsUpdate = true;
      for (const [k, c] of cubes) { disposeObj(c.mesh); cubes.delete(k); }   // re-lay at the new positions
    }
    if (view.phase === "complete" && phase !== "complete") celebrate();
    phase = view.phase;
    clearHints();
    if (!s) { setCubes({}); arm.aim(null, 1); return; }

    // Always the live board (SessionState.board), in every phase; the ready screen's thumbnail
    // shows the goal. Ready and complete turn the table slowly (not while the intro plays).
    setCubes(s.board);
    spinGoal = !intro.active && (phase === "ready" || phase === "complete") ? 1 : 0;

    let armTarget = null, armOpen = 1;
    if (phase === "running") {
      const e = s.expected;
      const cell = s.expectedCell || (e && e.cell);
      const level = s.expectedLevel ?? (e ? e.level : 0);
      if (cell && s.board[cell].length <= level) {
        const color = (s.finalBoard && s.finalBoard[cell] && s.finalBoard[cell][level]) || (e && e.color);
        [ghost, column] = makeGhost(color, cellPos(cell, level), s.mode === "human" && s.hintLevel >= 2);
      }
      if (view.faultVisible && s.faultCell) glows.push(makeGlow(s.faultCell));
      if (view.lift) lift = makeLift(cellPos(view.lift, Math.max(0, s.board[view.lift].length - 1)));
      if (view.arrow) {
        const [from, to] = view.arrow;
        const fromLevel = Math.max(0, s.board[from].length - 1);
        arrow = makeArrow(cellPos(from, fromLevel), cellPos(to, s.expectedLevel ?? 0));
      }
      const a = view.activity;
      if (a === "placing" || a === "fixing" || a === "pointing") {
        const tcell = a === "fixing" && s.mode === "robot" && s.faultCell ? s.faultCell : cell || s.faultCell;
        if (tcell) {
          const p = cellPos(tcell, s.board[tcell].length);
          p.y += a === "pointing" ? 1.1 : 0.55;
          armTarget = p;
          armOpen = a === "pointing" ? 0.2 : 0.8;
        }
      }
    }
    arm.aim(armTarget, armOpen);
  }

  // ------------------------------------------------------------ intro preview

  function introClear() {
    for (const h of intro.holos) disposeObj(h.mesh);
    intro.holos = [];
    intro.fading = 0;
    intro.active = false;
  }
  function introStart(viewSide) {
    introClear();
    if (viewSide && viewSide !== side) update({ s: null, phase, side: viewSide });
    intro.active = true;
    spinGoal = 0;   // square to the student while the build plays
  }
  function introDrop(color, cell, level) {
    if (!intro.active || !side) return;
    const mesh = makeHologram(color);
    const rest = cellPos(cell, level);
    mesh.position.copy(rest);
    mesh.renderOrder = 2;
    board.add(mesh);
    intro.holos.push({ mesh, rest: rest.y, born: performance.now() });
  }
  function introFade() { intro.fading = performance.now(); }

  // ------------------------------------------------------------ effects

  function pass(cell) {
    if (!cell || !side) return;
    const [x, z] = cellXZ(cell, side);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.52, 48),
      new THREE.MeshBasicMaterial({ color: OK_GREEN, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(x, 0.01, z);
    board.add(ring);
    pulses.push({ obj: ring, born: performance.now() });
  }

  function celebrate() {
    if (burst) disposeObj(burst.obj);
    const N = 360, posArr = new Float32Array(N * 3), colArr = new Float32Array(N * 3), vel = [];
    const palette = Object.values(CUBE_HEX).map((h) => new THREE.Color(h));
    for (let i = 0; i < N; i++) {
      posArr.set([0, 1.2, 0], i * 3);
      const a = Math.random() * Math.PI * 2, up = 3 + Math.random() * 3.5, out = 1 + Math.random() * 2.2;
      vel.push(new THREE.Vector3(Math.cos(a) * out, up, Math.sin(a) * out));
      const c = palette[i % palette.length];
      colArr.set([c.r, c.g, c.b], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.2, vertexColors: true, transparent: true, opacity: 1, depthWrite: false }));
    scene.add(pts);
    burst = { obj: pts, vel, born: performance.now() };
  }

  // ------------------------------------------------------------ frame loop

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);

  const easeOutBack = (t) => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2);

  function frame(tms) {
    if (!active) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.05, (tms - (last || tms)) / 1000);
    last = tms;
    const t = tms / 1000, now = performance.now();

    // Camera: student eye height, about 50 degrees down, with a very slow drift.
    const dist = 12.5, pitch = (50 * Math.PI) / 180;
    camera.position.set(Math.sin(t * 0.11) * 0.22, look.y + dist * Math.sin(pitch) + Math.sin(t * 0.07) * 0.1, look.z + dist * Math.cos(pitch));
    camera.lookAt(look);

    // Turntable on ready/complete, eased back to square when running.
    if (spinGoal) spin += dt * 0.25;
    else spin = Math.abs(spin % (Math.PI * 2)) < 0.002 ? 0 : spin + (Math.round(spin / (Math.PI * 2)) * Math.PI * 2 - spin) * (1 - Math.exp(-dt * 4));
    board.rotation.y = spin;
    arm.root.visible = !spinGoal;

    for (const [k, c] of cubes) {
      if (c.dying) {
        const f = Math.min(1, (now - c.dying) / FADE_MS);
        c.mesh.material.opacity = 1 - f;
        c.mesh.children[0].material.opacity = 1 - f;
        if (f >= 1) { disposeObj(c.mesh); cubes.delete(k); }
      } else if (c.born) {
        const f = Math.min(1, (now - c.born) / DROP_MS);
        c.mesh.position.y = c.rest + (1 - easeOutBack(f)) * 0.9;
        c.mesh.material.opacity = Math.min(1, f * 2.5);
        if (f >= 1) { c.born = 0; c.mesh.material.transparent = false; c.mesh.material.needsUpdate = true; }
      }
    }

    const still = reducedMotion();
    if (ghost) {   // drops into its slot, then shimmers; again every GHOST_CYCLE_MS
      const strong = ghost.userData.strong;
      const k = still ? 0.6 : 0.5 + 0.5 * Math.sin(t * (strong ? 5.5 : 3));
      const age = now - ghost.userData.born, cyc = age % GHOST_CYCLE_MS;
      const appear = still ? Math.min(1, age / HOLO_FADE_MS) : Math.min(1, cyc / 150);
      ghost.position.y = ghost.userData.base + (still ? 0 : dropOffset(cyc / HOLO_DROP_MS) * HOLO_DROP_H);
      ghost.material.opacity = appear * (strong ? 0.3 + 0.35 * k : 0.25 + 0.2 * k);
      ghost.children[0].material.opacity = appear * (0.55 + 0.45 * k);
      if (column) column.material.opacity = (strong ? 0.45 : 0.3) + 0.15 * k;
    }
    if (intro.holos.length) {
      const gone = intro.fading ? Math.min(1, (now - intro.fading) / HOLO_FADE_MS) : 0;
      for (const h of intro.holos) {
        const f = (now - h.born) / (still ? HOLO_FADE_MS : HOLO_DROP_MS);
        h.mesh.position.y = h.rest + (still ? 0 : dropOffset(f) * HOLO_DROP_H);
        const appear = Math.min(1, still ? f : f * 4);
        h.mesh.material.opacity = 0.45 * appear * (1 - gone);
        h.mesh.children[0].material.opacity = 0.9 * appear * (1 - gone);
      }
    }
    for (const g of glows) g.material.opacity = 0.5 + 0.25 * (0.5 + 0.5 * Math.sin(t * 2.4));
    if (arrow) arrow.userData.tex.offset.x -= dt * 1.4;
    if (lift) {   // rise and fade, again and again: "lift it off"
      const f = (t * 0.8) % 1;
      lift.position.y = lift.userData.base + 0.6 * f;
      lift.children.forEach((m) => { m.material.opacity = 0.95 * (1 - f * f); });
    }

    pulses = pulses.filter((p) => {
      const f = (now - p.born) / 800;
      if (f >= 1) { disposeObj(p.obj); return false; }
      p.obj.scale.setScalar(1 + f * 1.6);
      p.obj.material.opacity = 0.9 * (1 - f);
      return true;
    });

    if (burst) {
      const f = (now - burst.born) / 2600;
      if (f >= 1) { disposeObj(burst.obj); burst = null; }
      else {
        const arr = burst.obj.geometry.attributes.position.array;
        burst.vel.forEach((v, i) => {
          v.y -= 6.5 * dt;
          arr[i * 3] += v.x * dt; arr[i * 3 + 1] = Math.max(0.02, arr[i * 3 + 1] + v.y * dt); arr[i * 3 + 2] += v.z * dt;
        });
        burst.obj.geometry.attributes.position.needsUpdate = true;
        burst.obj.material.opacity = 1 - f * f;
      }
    }

    arm.step(dt);
    renderer.render(scene, camera);
  }

  function setActive(on) {
    if (on === active) return;
    active = on;
    last = 0;
    if (on) { resize(); raf = requestAnimationFrame(frame); } else cancelAnimationFrame(raf);
  }

  // Measured block colors ({ red: "#d8342f", ... }). Cubes already on the board recolor at once.
  function setColors(hexByColor) {
    for (const [color, hex] of Object.entries(hexByColor)) {
      if (!(color in CUBE_HEX) || !/^#[0-9a-f]{6}$/i.test(hex)) continue;
      CUBE_HEX[color] = parseInt(hex.slice(1), 16);
    }
    for (const cube of cubes.values()) {
      if (cube.color in CUBE_HEX) cube.mesh.material.color.setHex(CUBE_HEX[cube.color]);
    }
  }

  return { update, setActive, pass, setColors, introStart, introDrop, introFade, introClear };
}
