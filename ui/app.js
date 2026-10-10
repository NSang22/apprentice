// Apprentice monitor. Reads SessionState (README 5.4) from the brain on ws :8766 and draws it.
// Plain script, no dependencies: the 2D monitor must work even if nothing else loads.
// The 3D view (view3d.js, three.js) loads lazily and is optional.
(function () {
  "use strict";

  const CFG = window.APPRENTICE_CONFIG || {};
  const params = new URLSearchParams(location.search);
  const HUMAN_SIDE = params.get("side") || CFG.HUMAN_SIDE || "row3";
  const WS_URL = params.get("ws") || `ws://${location.hostname || "localhost"}:${CFG.BRAIN_WS_PORT || 8766}`;
  const PASS_HOLD_MS = CFG.PASS_HOLD_MS || 1600;

  const COLUMNS = ["A", "B", "C"];
  const ROWS = ["1", "2", "3"];
  const CELLS = COLUMNS.flatMap((c) => ROWS.map((r) => c + r));
  const COLORS = ["red", "blue", "yellow", "green"];
  const VERDICTS = ["PASS", "WRONG_CELL", "WRONG_COLOR", "WRONG_LEVEL", "EXTRA", "MISSING", "WAITING"];
  const LEVEL_WORDS = ["", "second", "third"];

  const $ = (id) => document.getElementById(id);
  const ui = {
    stage: $("stage"), lesson: $("lesson"), mode: $("mode"), conn: $("conn"), connText: $("conn-text"),
    banner: $("banner"), kicker: $("kicker"), instruction: $("instruction"), subline: $("subline"),
    feedback: $("feedback"), fbIcon: $("feedback-icon"), fbTitle: $("feedback-title"), fbText: $("feedback-text"),
    robotPanel: $("robot-panel"), robotStatus: $("robot-status"), robotText: $("robot-text"),
    showme: $("showme"), video: $("showme-video"),
    board2d: $("board2d"), board3d: $("board3d"), reconnect: $("reconnect"),
    stripHuman: $("strip-human"), stripRobot: $("strip-robot"),
  };

  const app = {
    conn: "connecting",   // connecting | live | reconnecting
    state: null,          // last normalized SessionState
    passUntil: 0,
    passText: null,
    boardKey: null,
    view: params.get("view") === "3d" ? "3d" : "2d",
    view3d: null,
    showKey: null,
    showFailed: null,
  };

  // ------------------------------------------------------------ input: normalize every field

  function cell(v) { return typeof v === "string" && CELLS.includes(v) ? v : null; }
  function str(v) { return typeof v === "string" && v.trim() ? v : null; }
  function num(v) { return typeof v === "number" && isFinite(v) ? v : null; }

  function normalize(d) {
    if (!d || typeof d !== "object" || (d.mode !== "human" && d.mode !== "robot")) return null;
    const board = {};
    for (const c of CELLS) {
      const stack = d.board && Array.isArray(d.board[c]) ? d.board[c] : [];
      board[c] = stack.filter((x) => typeof x === "string");
    }
    let expected = null;
    const e = d.expected_step;
    if (e && typeof e === "object" && cell(e.cell) && typeof e.color === "string") {
      expected = { color: e.color, cell: e.cell, level: Number.isInteger(e.level) ? e.level : 0 };
    }
    let showMe = null;
    const s = d.show_me;
    if (s && typeof s === "object" && str(s.video) && num(s.start) !== null && num(s.end) !== null && s.end > s.start) {
      showMe = { video: s.video, start: s.start, end: s.end };
    }
    const p = d.progress && typeof d.progress === "object" ? d.progress : {};
    const marks = (a) => (Array.isArray(a) ? a.map((x) => String(x)) : []);
    return {
      mode: d.mode,
      lessonId: str(d.lesson_id) || "",
      stepIndex: Number.isInteger(d.step_index) ? d.step_index : 0,
      board,
      expected,
      verdict: VERDICTS.includes(d.verdict) ? d.verdict : "WAITING",
      faultCell: cell(d.fault_cell),
      hintLevel: Number.isInteger(d.hint_level) ? Math.max(0, Math.min(4, d.hint_level)) : 0,
      hintText: str(d.hint_text),
      showMe,
      progress: { human: marks(p.human), robot: marks(p.robot) },
    };
  }

  // ------------------------------------------------------------ derived phase

  function phaseOf(s) {
    if (!s) return "nodata";
    if ((s.progress[s.mode] || []).includes("error")) return "stopped";
    if (!s.expected) return "complete";
    return "running";
  }

  function isMistake(s) {
    if (s.verdict === "PASS" || s.verdict === "WAITING") return false;
    return !(s.verdict === "MISSING" && !s.faultCell);  // MISSING with no fault cell = still placing
  }

  const MISTAKE_TITLES = {
    WRONG_CELL: "Almost! Wrong spot",
    WRONG_COLOR: "Almost! Different color",
    WRONG_LEVEL: "Right spot, different level",
    EXTRA: "One cube too many",
    MISSING: "A cube is missing",
  };

  function robotStatus(s, phase) {
    const t = (s.hintText || "").toLowerCase();
    if (phase === "stopped" || t.includes("needs a hand")) return { word: "Needs a hand", kind: "hand" };
    if (phase === "complete") return { word: "All done", kind: "done" };
    if (t.includes("getting ready")) return { word: "Getting ready", kind: "busy" };
    if (t.includes("taking")) return { word: "Fixing it", kind: "busy" };
    if (t.includes("placing")) return { word: "Placing", kind: "busy" };
    if (t.includes("caught a problem")) return { word: "Fixing it", kind: "busy" };
    if (s.verdict === "PASS") return { word: "Checks out", kind: "done" };
    return { word: "Watching", kind: "idle" };
  }

  // ------------------------------------------------------------ text helpers

  function esc(t) {
    return String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function chip(color) {
    const fill = COLORS.includes(color) ? `var(--cube-${color})` : "var(--cube-unknown)";
    return `<span class="chip" style="background:${fill}"></span>${esc(color)}`;
  }
  function stepSentence(step) {
    const verb = step.level > 0 ? "Stack" : "Place";
    return `${verb} the ${chip(step.color)} cube on <span class="cell">${esc(step.cell)}</span>`;
  }
  function robotSentence(step) {
    return `${chip(step.color)} cube to <span class="cell">${esc(step.cell)}</span>`.replace(/^(<span[^>]*><\/span>)(\w)/, (m, c, ch) => c + ch.toUpperCase());
  }
  function levelLine(step) {
    return step.level > 0 ? `On the ${LEVEL_WORDS[step.level] || "level " + (step.level + 1)} level` : "";
  }

  // ------------------------------------------------------------ render

  function render() {
    const s = app.state;
    const phase = phaseOf(s);
    ui.stage.dataset.phase = phase;

    // connection
    ui.conn.dataset.state = app.conn;
    ui.connText.textContent = app.conn === "live" ? "Live" : app.conn === "reconnecting" ? "Reconnecting" : "Connecting";
    const stale = app.conn !== "live" && s !== null;
    ui.stage.classList.toggle("stale", stale);
    ui.reconnect.hidden = !stale;

    guard(renderHeader, s);
    guard(renderLeft, s, phase);
    guard(renderBoard, s, phase);
    guard(renderStrips, s);
  }

  function guard(fn, ...args) {
    try { fn(...args); } catch (err) { console.error("[ui]", fn.name, err); }
  }

  function renderHeader(s) {
    ui.lesson.textContent = s ? s.lessonId : "";
    ui.mode.textContent = !s ? "" : s.mode === "robot" ? "Robot builds" : "You build";
  }

  function renderLeft(s, phase) {
    const now = Date.now();
    const total = s ? Math.max(s.progress.human.length, s.progress.robot.length) : 0;
    let banner = null, kicker = "", instruction = "", subline = "";
    let fb = { kind: "waiting", icon: "", title: "Waiting for the camera", text: "" };

    if (!s) {
      instruction = "Waiting for the camera";
      subline = app.conn === "live" ? "The lesson starts in a moment." : "Starting Apprentice...";
      fb = null;
    } else if (phase === "complete") {
      kicker = total ? `All ${total} steps` : "";
      instruction = s.mode === "robot" ? "The robot finished!" : "All done!";
      subline = "";
      fb = { kind: "complete", icon: "✓", title: s.hintText || "You built the whole thing.", text: "" };
    } else if (phase === "stopped") {
      kicker = `Step ${s.stepIndex + 1} of ${total}`;
      instruction = "The robot needs a hand";
      subline = "";
      const why = (s.hintText || "").replace(/^The robot needs a hand\.\s*/, "");
      fb = { kind: "stopped", icon: "!", title: "Let's sort it out together", text: why || "Please check the board, then restart the lesson." };
    } else {
      kicker = `${s.mode === "robot" ? "Robot step" : "Step"} ${s.stepIndex + 1} of ${total}`;
      instruction = s.mode === "robot" ? robotSentence(s.expected) : stepSentence(s.expected);
      subline = levelLine(s.expected);
      if (isMistake(s)) {
        app.passUntil = 0;
        fb = { kind: "mistake", icon: "!", title: MISTAKE_TITLES[s.verdict] || "Almost!", text: s.hintText || "" };
      } else if (now < app.passUntil) {
        fb = { kind: "pass", icon: "✓", title: app.passText || "Nice! That's right.", text: "" };
      } else if (s.verdict === "WAITING") {
        fb = { kind: "waiting", icon: "", title: "Waiting for the camera", text: "Hands off the board for a moment." };
      } else {
        fb = { kind: "watching", icon: "", title: "I'm watching the board", text: "" };
      }
      if (s.mode === "human" && s.hintLevel >= 4) banner = "The robot will point to the right spot.";
    }

    ui.banner.hidden = !banner;
    ui.banner.textContent = banner || "";
    ui.kicker.textContent = kicker;
    ui.instruction.innerHTML = instruction;
    ui.subline.textContent = subline;

    // robot mode: the robot panel carries what the robot is doing; the feedback card only celebrates
    const robot = s && s.mode === "robot";
    ui.robotPanel.hidden = !robot || phase !== "running";   // complete and stopped: the feedback card says it once
    if (robot) {
      const st = robotStatus(s, phase);
      ui.robotPanel.dataset.status = st.kind;
      ui.robotStatus.textContent = st.word;
      ui.robotText.textContent = s.hintText || "";
      if (fb.kind === "mistake" || fb.kind === "waiting" || fb.kind === "watching") fb = null;
    }

    ui.feedback.hidden = !fb;
    if (fb) {
      ui.feedback.dataset.kind = fb.kind;
      ui.fbIcon.textContent = fb.icon;
      ui.fbTitle.textContent = fb.title;
      ui.fbText.textContent = fb.text;
    }

    renderShowMe(s, phase);
  }

  // hint level 3: loop the expert clip between start and end, muted
  function renderShowMe(s, phase) {
    const want = s && s.mode === "human" && phase === "running" && s.hintLevel >= 3 && s.showMe;
    const key = want ? `${s.showMe.video}|${s.showMe.start}|${s.showMe.end}` : null;
    const showing = !!want && app.showFailed !== key;
    ui.stage.classList.toggle("showing", showing);
    if (!showing) {
      ui.showme.hidden = true;
      if (app.showKey) { ui.video.pause(); app.showKey = null; }
      return;
    }
    ui.showme.hidden = false;
    if (app.showKey === key) return;
    app.showKey = key;
    const v = ui.video;
    const src = /^(https?:)?\/\//.test(s.showMe.video) ? s.showMe.video : "/" + s.showMe.video.replace(/^\/+/, "");
    v.dataset.start = s.showMe.start;
    v.dataset.end = s.showMe.end;
    v.src = src;
    v.onloadedmetadata = () => { v.currentTime = s.showMe.start; v.play().catch(() => {}); };
    v.onerror = () => { app.showFailed = key; console.warn("[ui] show-me clip unavailable:", src); render(); };
  }
  ui.video.addEventListener("timeupdate", () => {
    const v = ui.video, start = Number(v.dataset.start), end = Number(v.dataset.end);
    if (isFinite(start) && isFinite(end) && (v.currentTime >= end || v.currentTime < start - 0.25)) v.currentTime = start;
  });
  ui.video.addEventListener("ended", () => { ui.video.currentTime = Number(ui.video.dataset.start) || 0; ui.video.play().catch(() => {}); });

  // ------------------------------------------------------------ board (2D)

  // Screen position of a cell as the student sees it: x left to right, y far to near.
  function screenXY(c) {
    const col = COLUMNS.indexOf(c[0]), row = ROWS.indexOf(c[1]);
    return HUMAN_SIDE === "row1" ? [2 - col, 2 - row] : [col, row];
  }

  const G = { x0: 70, y0: 70, pitch: 200, size: 184, inset: 22, step: 18 };

  function cellBox(c) {
    const [sx, sy] = screenXY(c);
    return { x: G.x0 + sx * G.pitch + 8, y: G.y0 + sy * G.pitch + 8 };
  }
  function levelRect(c, level) {
    const b = cellBox(c), i = G.inset + level * G.step;
    return { x: b.x + i, y: b.y + i, w: G.size - 2 * i };
  }
  function center(c) { const b = cellBox(c); return [b.x + G.size / 2, b.y + G.size / 2]; }
  function fill(color) { return COLORS.includes(color) ? `var(--cube-${color})` : "var(--cube-unknown)"; }

  function arrowFrom(s) {
    // Hint level 2+: arrow from the misplaced cube to where it goes. Only when the fault cube is
    // clearly the current step's cube (SessionState has no expected_cell for older cubes).
    if (s.mode !== "human" || s.hintLevel < 2 || s.verdict !== "WRONG_CELL" || !s.faultCell || !s.expected) return null;
    if (s.faultCell === s.expected.cell) return null;
    const stack = s.board[s.faultCell];
    if (!stack.length || stack[stack.length - 1] !== s.expected.color) return null;
    return [s.faultCell, s.expected.cell];
  }

  function renderBoard(s, phase) {
    const key = JSON.stringify(s ? [s.board, s.expected, s.faultCell, s.hintLevel, s.verdict, phase, HUMAN_SIDE] : null);
    if (key === app.boardKey) return;
    app.boardKey = key;
    const faultVisible = !!(s && phase === "running" && s.faultCell &&
      (s.mode === "human" ? isMistake(s) : s.verdict !== "PASS" && s.verdict !== "WAITING"));
    if (app.view3d && app.view === "3d") guard(app.view3d.update, s, phase, HUMAN_SIDE, faultVisible);

    const W = G.x0 * 2 + G.pitch * 3 - 16, H = G.y0 + G.pitch * 3 + 90;
    const parts = [`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Board">`];
    parts.push(`<defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path class="b-arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>`);

    const gridW = G.pitch * 3 - 16;
    parts.push(`<text class="b-edge" x="${G.x0 + gridW / 2}" y="${G.y0 - 30}" text-anchor="middle">Robot side</text>`);
    parts.push(`<text class="b-edge" x="${G.x0 + gridW / 2}" y="${G.y0 + G.pitch * 3 + 72}" text-anchor="middle">Your side</text>`);
    for (const col of COLUMNS) {
      const [sx] = screenXY(col + "1");
      parts.push(`<text class="b-axis" x="${G.x0 + sx * G.pitch + G.size / 2 + 8}" y="${G.y0 + G.pitch * 3 + 26}" text-anchor="middle">${col}</text>`);
    }
    for (const row of ROWS) {
      const [, sy] = screenXY("A" + row);
      parts.push(`<text class="b-axis" x="${G.x0 - 34}" y="${G.y0 + sy * G.pitch + G.size / 2 + 20}" text-anchor="middle">${row}</text>`);
    }

    for (const c of CELLS) {
      const b = cellBox(c);
      parts.push(`<rect class="b-cell" x="${b.x}" y="${b.y}" width="${G.size}" height="${G.size}" rx="16"/>`);
      parts.push(`<text class="b-name" x="${b.x + 8}" y="${b.y + 19}">${c}</text>`);
      const stack = s ? s.board[c] : [];
      stack.forEach((color, level) => {
        const r = levelRect(c, level);
        parts.push(`<rect class="b-cube" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.w}" rx="10" style="fill:${fill(color)}"/>`);
        if (!COLORS.includes(color)) parts.push(`<text class="b-unknown" x="${r.x + r.w / 2}" y="${r.y + r.w / 2 + 14}" text-anchor="middle">?</text>`);
      });
    }

    if (s && phase === "running") {
      if (faultVisible) {
        const b = cellBox(s.faultCell);
        parts.push(`<rect class="b-fault" x="${b.x - 6}" y="${b.y - 6}" width="${G.size + 12}" height="${G.size + 12}" rx="20"/>`);
      }
      const e = s.expected;
      if (e && s.board[e.cell].length <= e.level) {   // only while its slot is still open
        const r = levelRect(e.cell, Math.min(e.level, 2));
        const strong = s.mode === "human" && s.hintLevel >= 2 ? " strong" : "";
        parts.push(`<rect class="b-ghost${strong}" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.w}" rx="10" style="fill:${fill(e.color)}"/>`);
      }
      const arrow = arrowFrom(s);
      if (arrow) {
        const [x1, y1] = center(arrow[0]), [x2, y2] = center(arrow[1]);
        const d = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / d, uy = (y2 - y1) / d, pad = 70;
        parts.push(`<line class="b-arrow" x1="${x1 + ux * pad}" y1="${y1 + uy * pad}" x2="${x2 - ux * pad}" y2="${y2 - uy * pad}" marker-end="url(#ah)"/>`);
      }
    }
    parts.push("</svg>");
    ui.board2d.innerHTML = parts.join("");
  }

  // ------------------------------------------------------------ progress strips

  function renderStrips(s) {
    const n = s ? Math.max(s.progress.human.length, s.progress.robot.length) : 0;
    fillStrip(ui.stripHuman, s ? s.progress.human : [], n);
    fillStrip(ui.stripRobot, s ? s.progress.robot : [], n);
  }
  function fillStrip(el, marks, n) {
    el.classList.toggle("off", marks.length === 0);
    const html = [];
    for (let i = 0; i < n; i++) {
      const m = ["pass", "retry", "error", "active"].includes(marks[i]) ? marks[i] : "pending";
      html.push(`<div class="seg ${m}">${i + 1}</div>`);
    }
    const out = html.join("");
    if (el.innerHTML !== out) el.innerHTML = out;
  }

  // ------------------------------------------------------------ connection

  let backoff = CFG.RECONNECT_MIN_MS || 500;
  function connect() {
    let ws;
    try { ws = new WebSocket(WS_URL); } catch (err) { return retry(); }
    ws.onopen = () => { app.conn = "live"; backoff = CFG.RECONNECT_MIN_MS || 500; render(); };
    ws.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch (err) { return; }
      const s = normalize(d);
      if (!s) return;
      onState(s);
    };
    ws.onclose = () => { app.conn = app.state ? "reconnecting" : "connecting"; render(); retry(); };
    ws.onerror = () => { try { ws.close(); } catch (err) { /* closing anyway */ } };
  }
  function retry() {
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 1.5, CFG.RECONNECT_MAX_MS || 3000);
  }

  function onState(s) {
    const prev = app.state;
    if (prev && prev.lessonId !== s.lessonId) app.passUntil = 0;   // brain restarted on another lesson
    if (s.verdict === "PASS") {
      app.passUntil = Date.now() + PASS_HOLD_MS;
      app.passText = s.hintText || "Nice! That's right.";
      setTimeout(render, PASS_HOLD_MS + 50);
    }
    app.state = s;
    render();
  }

  // ------------------------------------------------------------ 3D toggle (touch free: keys 2 / 3 / v, or ?view=3d)

  async function setView(view) {
    if (view === "3d" && !app.view3d) {
      try {
        const mod = await import("./view3d.js");
        app.view3d = mod.createView3D(ui.board3d);
      } catch (err) {
        console.warn("[ui] 3D view unavailable, staying in 2D:", err);
        app.view = "2d";
        ui.board3d.hidden = true;
        ui.board2d.hidden = false;
        return;
      }
    }
    app.view = view;
    ui.board2d.hidden = view !== "2d";
    ui.board3d.hidden = view !== "3d";
    if (app.view3d) app.view3d.setActive(view === "3d");
    app.boardKey = null;
    render();
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "3") setView("3d");
    else if (ev.key === "2") setView("2d");
    else if (ev.key === "v") setView(app.view === "3d" ? "2d" : "3d");
  });

  render();
  connect();
  if (app.view === "3d") setView("3d");
})();
