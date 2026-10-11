// Apprentice monitor. Reads SessionState (README 5.4) from the brain on ws :8766 and draws it.
// Plain script, no dependencies: the 2D monitor must work even if nothing else loads.
// The 3D view (view3d.js, three.js) is the main view; it loads lazily, and the 2D board takes
// over (key 2, or automatically) if WebGL or the module is unavailable.
// Keys: Space plays the intro preview on the ready screen, then sends {"cmd":"start"} (Space again
// skips the intro). R sends {"cmd":"reset"}. L opens the lesson picker ({"cmd":"list"} every 3 s,
// Enter sends {"cmd":"load"}). B on a finished human lesson sends {"cmd":"mode","mode":"robot"}.
// 2 / 3 switch views.
(function () {
  "use strict";

  const CFG = window.APPRENTICE_CONFIG || {};
  const params = new URLSearchParams(location.search);
  const WS_URL = params.get("ws") || `ws://${location.hostname || "localhost"}:${CFG.BRAIN_WS_PORT || 8766}`;
  const PASS_HOLD_MS = CFG.PASS_HOLD_MS || 1600;
  const INTRO_STEP_MS = CFG.INTRO_STEP_MS || 800;
  const INTRO_HOLD_MS = CFG.INTRO_HOLD_MS || 1200;
  const INTRO_DROP_MS = 600, INTRO_FADE_MS = 500;   // match view3d.js
  const LIST_POLL_MS = 3000;
  const NEW_LESSON_S = 600;                          // a lesson file this recent is marked "new"

  // Cells follow the printed sheet: the letter is the row (A along the robot's edge, C along the
  // student's), the number is the column (1 on the left for a student opposite the robot).
  const ROW_LETTERS = ["A", "B", "C"];
  const COLUMN_NUMBERS = ["1", "2", "3"];
  const CELLS = ROW_LETTERS.flatMap((r) => COLUMN_NUMBERS.map((c) => r + c));
  const COLORS = ["red", "orange", "yellow", "teal", "skyblue", "purple"];   // shared/config.py COLORS
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
    screen: $("screen"), screenTitle: $("screen-title"), screenThumb: $("screen-thumb"),
    screenText: $("screen-text"), screenKey: $("screen-key"), toast: $("toast"),
    screenAction: $("screen-action"), picker: $("picker"), pickerList: $("picker-list"),
  };

  const app = {
    conn: "connecting",   // connecting | live | reconnecting
    state: null,          // last normalized SessionState
    passUntil: 0,
    passText: null,
    boardKey: null,
    view: params.get("view") === "2d" ? "2d" : "3d",
    view3d: null,
    showKey: null,
    showFailed: null,
    ws: null,
    toastTimer: 0,
    blockHex: {},         // color -> measured hex from perception/calib/colors.json
    intro: null,          // { steps, shown, fading, timers } while the intro preview plays
    autoIntro: false,     // play the intro once when the lesson we asked for arrives in "ready"
    turnText: null,       // "Your turn! ..." banner right after the intro
    turnUntil: 0,
    picker: { open: false, items: [], sel: 0, timer: 0 },
  };

  // The student's side comes from the brain (SessionState.human_side); "row3" (opposite the robot,
  // at the C-row edge) until the first state.
  function side() { return (app.state && app.state.humanSide) || "row3"; }

  // ------------------------------------------------------------ input: normalize every field

  function cell(v) { return typeof v === "string" && CELLS.includes(v) ? v : null; }
  function str(v) { return typeof v === "string" && v.trim() ? v : null; }
  function num(v) { return typeof v === "number" && isFinite(v) ? v : null; }

  function boardOf(b) {
    const board = {};
    for (const c of CELLS) {
      const stack = b && Array.isArray(b[c]) ? b[c] : [];
      board[c] = stack.filter((x) => typeof x === "string");
    }
    return board;
  }
  function isClear(board) { return CELLS.every((c) => board[c].length === 0); }

  function normalize(d) {
    if (!d || typeof d !== "object" || (d.mode !== "human" && d.mode !== "robot")) return null;
    const board = boardOf(d.board);
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
      phase: str(d.phase),                          // brain's phase; enums live in shared/schemas.py
      lessonTitle: str(d.lesson_title),
      finalBoard: d.final_board && typeof d.final_board === "object" ? boardOf(d.final_board) : null,
      humanSide: d.human_side === "row1" || d.human_side === "row3" ? d.human_side : null,
      expectedCell: cell(d.expected_cell),
      expectedLevel: Number.isInteger(d.expected_level) ? d.expected_level : null,
      robotStatus: str(d.robot_status) || "idle",
      boardSeen: d.board_seen === true,              // false until the camera produced a stable board
      lessonSteps: Array.isArray(d.lesson_steps)     // the whole lesson in order, for the intro preview
        ? d.lesson_steps.filter((x) => x && cell(x.cell) && typeof x.color === "string")
            .map((x) => ({ color: x.color, cell: x.cell, level: Number.isInteger(x.level) ? x.level : 0, narration: str(x.narration) }))
        : null,
    };
  }

  // ------------------------------------------------------------ derived phase

  function phaseOf(s) {
    if (!s) return "nodata";
    if (s.phase) return s.phase;
    if ((s.progress[s.mode] || []).includes("error")) return "stopped";   // older brains: derive it
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
    EXTRA: "One block too many",
    MISSING: "A cube is missing",
  };

  function robotStatus(s, phase) {
    const a = s.robotStatus;
    if (phase === "stopped" || a === "stopped") return { word: "Needs a hand", kind: "hand" };
    if (phase === "complete") return { word: "All done", kind: "done" };
    if (a === "placing") return { word: "Placing", kind: "busy" };
    if (a === "fixing") return { word: "Fixing it", kind: "busy" };
    if (a === "pointing") return { word: "Pointing", kind: "busy" };
    if (a === "homing") return { word: "Going home", kind: "busy" };
    if (s.verdict === "PASS") return { word: "Checks out", kind: "done" };
    if (a === "checking") return { word: "Checking", kind: "idle" };
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
  function stepText(step) {   // the intro headline: the lesson's own words, else "Red on B2"
    return step.narration || `${step.color[0].toUpperCase()}${step.color.slice(1)} on ${step.cell}`;
  }
  function levelLine(step) {
    return step.level > 0 ? `On the ${LEVEL_WORDS[step.level] || "level " + (step.level + 1)} level` : "";
  }

  // ------------------------------------------------------------ render

  function render() {
    const s = app.state;
    const phase = phaseOf(s);
    ui.stage.dataset.phase = phase;
    ui.stage.classList.toggle("intro", !!app.intro);

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
    guard(renderScreen, s, phase);
    guard(renderPicker, s);
  }

  function guard(fn, ...args) {
    try { fn(...args); } catch (err) { console.error("[ui]", fn.name, err); }
  }

  function renderHeader(s) {
    ui.lesson.textContent = s ? s.lessonTitle || s.lessonId : "";
    ui.mode.textContent = !s ? "" : s.mode === "robot" ? "Robot builds" : "You build";
  }

  function renderLeft(s, phase) {
    const now = Date.now();
    const total = s ? Math.max(s.progress.human.length, s.progress.robot.length) : 0;
    let banner = null, kicker = "", instruction = "", subline = "";
    let fb = { kind: "waiting", icon: "", title: "Waiting for the camera", text: "" };

    if (app.intro) {
      const n = app.intro.steps.length, k = app.intro.shown;
      kicker = k ? `Preview: step ${k} of ${n}` : "Here is the whole build";
      instruction = k ? esc(stepText(app.intro.steps[k - 1])) : esc(s ? s.lessonTitle || "Watch" : "Watch");
      subline = "Press Space to skip";
      fb = null;
    } else if (!s) {
      instruction = "Waiting for the camera";
      subline = app.conn === "live" ? "The lesson starts in a moment." : "Starting Apprentice...";
      fb = null;
    } else if (phase === "ready") {
      kicker = total ? `${total} steps` : "";
      instruction = s.lessonTitle || "Get ready";
      fb = null;   // the ready screen carries hint_text and the start prompt
    } else if (phase === "complete") {
      kicker = total ? `All ${total} steps` : "";
      instruction = s.mode === "robot" ? "The robot finished!" : "All done!";
      subline = "";
      fb = { kind: "complete", icon: "✓", title: s.hintText || "You built the whole thing.", text: "" };
    } else if (phase === "stopped") {
      kicker = `Step ${s.stepIndex + 1} of ${total}`;
      instruction = "The robot needs a hand";
      subline = "";
      fb = null;   // the stopped screen says it once, calmly
    } else {
      kicker = `${s.mode === "robot" ? "Robot step" : "Step"} ${s.stepIndex + 1} of ${total}`;
      instruction = s.mode === "robot" ? robotSentence(s.expected) : stepSentence(s.expected);
      subline = levelLine(s.expected);
      if (isMistake(s)) {
        app.passUntil = 0;
        fb = { kind: "mistake", icon: "!", title: MISTAKE_TITLES[s.verdict] || "Almost!", text: s.hintText || "" };
      } else if (unsureCell(s)) {
        app.passUntil = 0;
        fb = { kind: "unsure", icon: "?", title: "I can't read one block", text: s.hintText || "" };
      } else if (now < app.passUntil) {
        fb = { kind: "pass", icon: "✓", title: app.passText || "Nice! That's right.", text: "" };
      } else if (!s.boardSeen) {
        fb = { kind: "waiting", icon: "", title: "Waiting for the camera", text: "Hands off the board for a moment." };
      } else {
        fb = { kind: "watching", icon: "", title: "I'm watching the board", text: "" };
      }
      if (s.robotStatus === "pointing") banner = "The robot is pointing to the right spot.";
      else if (now < app.turnUntil) banner = app.turnText;
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
      if (fb && (fb.kind === "mistake" || fb.kind === "waiting" || fb.kind === "watching")) fb = null;
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
  function screenXY(c, sd = side()) {
    const col = COLUMN_NUMBERS.indexOf(c[1]), row = ROW_LETTERS.indexOf(c[0]);
    return sd === "row1" ? [2 - col, 2 - row] : [col, row];
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
    // Hint level 2+: arrow from the misplaced block (fault_cell) to the slot that needs it
    // (expected_cell, from the checker). Only WRONG_CELL moves a block to another cell; EXTRA
    // means "take it off" and gets the lift marker instead (liftFrom), never an arrow.
    if (s.mode !== "human" || s.hintLevel < 2 || s.verdict !== "WRONG_CELL" || !s.faultCell || !s.expectedCell) return null;
    if (s.faultCell === s.expectedCell) return null;
    return [s.faultCell, s.expectedCell];
  }

  // EXTRA: the block on fault_cell comes off the board. Shown from the first hint on.
  function liftFrom(s, phase) {
    return s && phase === "running" && s.verdict === "EXTRA" && s.faultCell && s.board[s.faultCell].length ? s.faultCell : null;
  }

  // WAITING with a fault cell: the camera can't read that block's color.
  function unsureCell(s) {
    return s && s.verdict === "WAITING" && s.faultCell ? s.faultCell : null;
  }

  function renderBoard(s, phase) {
    const intro = app.intro ? app.intro.shown : -1;
    const key = JSON.stringify(s ? [s.board, s.expected, s.stepIndex, s.faultCell, s.expectedCell, s.hintLevel, s.verdict, phase, side(), s.robotStatus, intro] : null);
    if (key === app.boardKey) return;
    app.boardKey = key;
    const faultVisible = !!(s && phase === "running" && s.faultCell &&
      (s.mode === "human" ? isMistake(s) : s.verdict !== "PASS" && s.verdict !== "WAITING"));
    if (app.view3d && app.view === "3d") {
      guard(app.view3d.update, { s, phase, side: side(), faultVisible, arrow: s ? arrowFrom(s) : null,
                                 lift: liftFrom(s, phase), unsure: unsureCell(s), activity: s ? s.robotStatus : null });
    }

    const W = G.x0 * 2 + G.pitch * 3 - 16, H = G.y0 + G.pitch * 3 + 90;
    const parts = [`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Board">`];
    parts.push(`<defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path class="b-arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>`);

    const gridW = G.pitch * 3 - 16;
    parts.push(`<text class="b-edge" x="${G.x0 + gridW / 2}" y="${G.y0 - 30}" text-anchor="middle">Robot side</text>`);
    parts.push(`<text class="b-edge" x="${G.x0 + gridW / 2}" y="${G.y0 + G.pitch * 3 + 72}" text-anchor="middle">Your side</text>`);
    for (const col of COLUMN_NUMBERS) {   // column numbers along the student's edge
      const [sx] = screenXY("A" + col);
      parts.push(`<text class="b-axis" x="${G.x0 + sx * G.pitch + G.size / 2 + 8}" y="${G.y0 + G.pitch * 3 + 26}" text-anchor="middle">${col}</text>`);
    }
    for (const row of ROW_LETTERS) {      // row letters down the side, A at the robot's edge
      const [, sy] = screenXY(row + "1");
      parts.push(`<text class="b-axis" x="${G.x0 - 34}" y="${G.y0 + sy * G.pitch + G.size / 2 + 20}" text-anchor="middle">${row}</text>`);
    }

    for (const c of CELLS) {
      const b = cellBox(c);
      parts.push(`<rect class="b-cell" x="${b.x}" y="${b.y}" width="${G.size}" height="${G.size}" rx="16"/>`);
      parts.push(`<text class="b-name" x="${b.x + 8}" y="${b.y + 19}">${c}</text>`);
      const stack = app.intro ? [] : s ? s.board[c] : [];
      stack.forEach((color, level) => {
        const r = levelRect(c, level);
        parts.push(`<rect class="b-cube" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.w}" rx="10" style="fill:${fill(color)}"/>`);
        if (!COLORS.includes(color)) parts.push(`<text class="b-unknown" x="${r.x + r.w / 2}" y="${r.y + r.w / 2 + 14}" text-anchor="middle">?</text>`);
      });
    }

    if (app.intro) {   // the intro preview: the lesson's cubes fill in, in order, as ghosts
      app.intro.steps.slice(0, app.intro.shown).forEach((st) => {
        const r = levelRect(st.cell, Math.min(st.level, 2));
        parts.push(`<rect class="b-intro" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.w}" rx="10" style="fill:${fill(st.color)}"/>`);
      });
    }

    if (s && phase === "running") {
      if (faultVisible) {
        const b = cellBox(s.faultCell);
        parts.push(`<rect class="b-fault" x="${b.x - 6}" y="${b.y - 6}" width="${G.size + 12}" height="${G.size + 12}" rx="20"/>`);
      }
      const e = s.expected;
      const gCell = s.expectedCell || (e && e.cell);
      const gLevel = s.expectedLevel ?? (e ? e.level : 0);
      const gColor = (s.finalBoard && s.finalBoard[gCell] && s.finalBoard[gCell][gLevel]) || (e && e.color);
      if (gCell && s.board[gCell].length <= gLevel) {   // only while its slot is still open
        const r = levelRect(gCell, Math.min(gLevel, 2));
        const strong = s.mode === "human" && s.hintLevel >= 2 ? " strong" : "";
        parts.push(`<rect class="b-ghost${strong}" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.w}" rx="10" style="fill:${fill(gColor)}"/>`);
      }
      const lift = liftFrom(s, phase);
      if (lift) {   // take it off: dashed outline on the block and a "lift off" badge, no arrow
        const b = cellBox(lift), top = levelRect(lift, s.board[lift].length - 1);
        parts.push(`<rect class="b-remove" x="${top.x - 6}" y="${top.y - 6}" width="${top.w + 12}" height="${top.w + 12}" rx="14"/>`);
        parts.push(`<g class="b-lift"><rect x="${b.x + G.size / 2 - 62}" y="${b.y + G.size - 54}" width="124" height="44" rx="22"/>` +
                   `<text x="${b.x + G.size / 2}" y="${b.y + G.size - 23}" text-anchor="middle">\u2191 off</text></g>`);
      }
      const unsure = unsureCell(s);
      if (unsure) {
        const b = cellBox(unsure);
        parts.push(`<rect class="b-unsure" x="${b.x - 6}" y="${b.y - 6}" width="${G.size + 12}" height="${G.size + 12}" rx="20"/>`);
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
    app.ws = ws;
    ws.onopen = () => { app.conn = "live"; backoff = CFG.RECONNECT_MIN_MS || 500; render(); };
    ws.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch (err) { return; }
      if (d && Array.isArray(d.lessons)) return onLessons(d.lessons);   // reply to {"cmd":"list"}
      if (d && typeof d.error === "string") { app.autoIntro = false; return toast(d.error, 4000); }
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
    if (prev && s.stepIndex < prev.stepIndex) app.passUntil = 0;     // a block came off: show the earlier step now
    if (s.verdict === "PASS") {
      if (app.view3d && app.view === "3d") guard(app.view3d.pass, s.expectedCell || (s.expected && s.expected.cell));
      app.passUntil = Date.now() + PASS_HOLD_MS;
      app.passText = s.hintText || "Nice! That's right.";
      setTimeout(render, PASS_HOLD_MS + 50);
    }
    app.state = s;
    if (app.intro && phaseOf(s) !== "ready") cancelIntro();   // started or reset elsewhere
    if (app.autoIntro && phaseOf(s) === "ready" && s.lessonSteps && s.lessonSteps.length) {
      app.autoIntro = false;   // a lesson we just loaded (or the robot's turn): show the whole build once
      playIntro();
      return;
    }
    render();
  }

  // ------------------------------------------------------------ 3D toggle (touch free: keys 2 / 3 / v, or ?view=3d)

  async function setView(view) {
    if (view === "3d" && !app.view3d) {
      try {
        const mod = await import("./view3d.js");
        app.view3d = mod.createView3D(ui.board3d);
        if (Object.keys(app.blockHex).length) guard(app.view3d.setColors, app.blockHex);
      } catch (err) {
        console.warn("[ui] 3D view unavailable, using the 2D board:", err);
        app.view3dFailed = true;
        view = "2d";
      }
    }
    if (view === "3d" && app.view3dFailed) view = "2d";
    app.view = view;
    ui.stage.classList.toggle("view-3d", view === "3d");
    ui.board2d.hidden = view !== "2d";
    ui.board3d.hidden = view !== "3d";
    if (app.view3d) app.view3d.setActive(view === "3d");
    app.boardKey = null;
    render();
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.repeat) return;
    if (app.picker.open && pickerKey(ev)) return;
    if (ev.key === "l" || ev.key === "L") { openPicker(); return; }
    if (ev.key === "b" || ev.key === "B") { nowTheRobot(); return; }
    if (ev.key === "3") setView("3d");
    else if (ev.key === "2") setView("2d");
    else if (ev.key === "v") setView(app.view === "3d" ? "2d" : "3d");
    else if (ev.key === " " || ev.code === "Space") { ev.preventDefault(); spaceKey(); }
    else if (ev.key === "r" || ev.key === "R") resetLesson();
  });

  // ------------------------------------------------------------ commands to the brain (ws :8766)

  function send(cmd, extra = {}, quiet = false) {
    const ws = app.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) { if (!quiet) toast("Not connected to Apprentice yet"); return false; }
    ws.send(JSON.stringify({ cmd, ...extra }));
    return true;
  }
  function spaceKey() {
    if (app.intro) return finishIntro();   // Space during the intro skips it
    const s = app.state;
    if (!s || phaseOf(s) !== "ready") return;
    if (s.lessonSteps && s.lessonSteps.length && isClear(s.board)) playIntro();
    else startLesson();
  }
  function startLesson() {
    const s = app.state;
    if (!s || phaseOf(s) !== "ready") return;
    if (!isClear(s.board)) { toast("Please clear the board first"); return; }
    if (send("start")) toast("Starting...");
  }
  function resetLesson() {
    cancelIntro();
    if (send("reset")) toast("Starting over...");
  }
  function toast(text, ms = 1600) {
    ui.toast.textContent = text;
    ui.toast.hidden = false;
    clearTimeout(app.toastTimer);
    app.toastTimer = setTimeout(() => { ui.toast.hidden = true; }, ms);
  }

  // ------------------------------------------------------------ intro preview (ready screen)

  // Each step's cube drops in as a hologram, one every INTRO_STEP_MS, with that step's words as
  // the headline. The finished hologram holds, fades, and then the lesson starts.
  function playIntro() {
    const s = app.state;
    if (!s || !s.lessonSteps || !s.lessonSteps.length) return startLesson();
    cancelIntro();
    const steps = s.lessonSteps, n = steps.length;
    const intro = { steps, shown: 0, timers: [] };
    app.intro = intro;
    if (app.view3d) guard(app.view3d.introStart, side());
    steps.forEach((st, i) => intro.timers.push(setTimeout(() => {
      intro.shown = i + 1;
      if (app.view3d) guard(app.view3d.introDrop, st.color, st.cell, st.level);
      render();
    }, i * INTRO_STEP_MS)));
    const fadeAt = (n - 1) * INTRO_STEP_MS + INTRO_DROP_MS + INTRO_HOLD_MS;
    intro.timers.push(setTimeout(() => { if (app.view3d) guard(app.view3d.introFade); }, fadeAt));
    intro.timers.push(setTimeout(finishIntro, fadeAt + INTRO_FADE_MS));
    app.boardKey = null;
    render();
  }
  function cancelIntro() {
    if (!app.intro) return;
    app.intro.timers.forEach(clearTimeout);
    app.intro = null;
    if (app.view3d) guard(app.view3d.introClear);
    app.boardKey = null;
  }
  function finishIntro() {
    const steps = app.intro ? app.intro.steps : null;
    cancelIntro();
    const s = app.state;
    if (steps && s && phaseOf(s) === "ready" && send("start")) {
      app.turnText = (s.mode === "robot" ? "The robot's turn! " : "Your turn! ") + stepText(steps[0]);
      app.turnUntil = Date.now() + 3500;
      setTimeout(render, 3550);
    }
    render();
  }

  // ------------------------------------------------------------ lesson picker (L) and the robot's turn (B)

  function openPicker() {
    if (app.picker.open) return closePicker();
    app.picker.open = true;
    send("list");
    app.picker.timer = setInterval(() => send("list", {}, true), LIST_POLL_MS);
    render();
  }
  function closePicker() {
    app.picker.open = false;
    clearInterval(app.picker.timer);
    render();
  }
  function onLessons(list) {
    const items = list.filter((d) => d && typeof d.lesson_id === "string")
      .map((d) => ({ id: d.lesson_id, title: str(d.title) || d.lesson_id, steps: Number(d.steps) || 0, mtime: Number(d.mtime) || 0 }));
    const was = app.picker.items[app.picker.sel];
    app.picker.items = items;   // the brain sends them newest first
    const keep = was ? items.findIndex((d) => d.id === was.id) : -1;
    app.picker.sel = keep >= 0 ? keep : 0;
    render();
  }
  function pickerKey(ev) {
    const p = app.picker, n = p.items.length;
    if (ev.key === "ArrowDown") { p.sel = n ? (p.sel + 1) % n : 0; }
    else if (ev.key === "ArrowUp") { p.sel = n ? (p.sel - 1 + n) % n : 0; }
    else if (ev.key === "Enter") { if (n) loadLesson(p.items[p.sel].id); return true; }
    else if (ev.key === "Escape" || ev.key === "l" || ev.key === "L") { closePicker(); return true; }
    else return false;
    ev.preventDefault();
    render();
    return true;
  }
  function loadLesson(id) {
    cancelIntro();
    closePicker();
    if (send("load", { lesson_id: id })) { app.autoIntro = true; toast("Loading the lesson..."); }
  }
  function renderPicker(s) {
    ui.picker.hidden = !app.picker.open;
    if (!app.picker.open) return;
    const now = Date.now() / 1000;
    const rows = app.picker.items.map((d, i) => {
      const tags = [d.mtime && now - d.mtime < NEW_LESSON_S ? `<span class="tag new">new</span>` : "",
                    s && s.lessonId === d.id ? `<span class="tag now">now</span>` : ""].join("");
      return `<li class="${i === app.picker.sel ? "sel" : ""}" data-id="${esc(d.id)}"><span class="p-title">${esc(d.title)}</span>` +
             `<span class="p-meta">${d.steps} steps</span>${tags}</li>`;
    });
    const html = rows.length ? rows.join("") : `<li class="empty">Looking for lessons...</li>`;
    if (ui.pickerList.innerHTML !== html) ui.pickerList.innerHTML = html;
  }
  ui.pickerList.addEventListener("click", (ev) => {
    const li = ev.target.closest("li[data-id]");
    if (li) loadLesson(li.dataset.id);
  });

  function nowTheRobot() {
    const s = app.state;
    if (!s || s.mode !== "human" || phaseOf(s) !== "complete") return;
    if (send("mode", { mode: "robot" })) { app.autoIntro = true; toast("Getting the robot ready..."); }
  }
  ui.screenAction.addEventListener("click", nowTheRobot);

  // ------------------------------------------------------------ ready / complete / stopped screens

  function thumbSVG(board, sd) {
    // Small top-down picture of a board from the student's side; stacks show as layered squares.
    const P = 64, S = 56, parts = [`<svg viewBox="0 0 ${P * 3 + 8} ${P * 3 + 8}" xmlns="http://www.w3.org/2000/svg">`];
    for (const c of CELLS) {
      const [sx, sy] = screenXY(c, sd);
      const x = 4 + sx * P + (P - S) / 2, y = 4 + sy * P + (P - S) / 2;
      parts.push(`<rect class="t-cell" x="${x}" y="${y}" width="${S}" height="${S}" rx="8"/>`);
      (board[c] || []).forEach((color, level) => {
        const i = 6 + level * 7;
        parts.push(`<rect class="t-cube" x="${x + i}" y="${y + i}" width="${S - 2 * i}" height="${S - 2 * i}" rx="6" style="fill:${fill(color)}"/>`);
      });
    }
    parts.push("</svg>");
    return parts.join("");
  }

  function renderScreen(s, phase) {
    let show = s && !app.intro && (phase === "ready" || phase === "complete" || phase === "stopped");
    ui.screen.hidden = !show;
    ui.stage.classList.toggle("screened", !!show);
    if (!show) return;
    ui.screen.dataset.kind = phase;
    let title = "", text = "", key = "", board = null;
    if (phase === "ready") {
      title = s.lessonTitle || "Ready";
      board = s.finalBoard;
      const camera = !s.boardSeen;
      text = camera ? "Waiting for the camera" : isClear(s.board) ? (s.mode === "robot" ? "The robot is ready to build this." : "Build this, one cube at a time.") : (s.hintText || "Please clear the board to start.");
      key = !camera && isClear(s.board) ? "Press Space to start" : "";
    } else if (phase === "complete") {
      title = "All done!";
      board = s.board;
      text = (s.hintText || "").replace(/^All done!\s*/, "");   // the title already says it
      key = "Press R to start again";
    } else {
      title = "Let's sort it out together";
      text = s.hintText || "The robot needs a hand. Please check the board.";
      key = "Press R to try again";
    }
    ui.screenTitle.textContent = title;
    ui.screenText.textContent = text;
    ui.screenKey.textContent = key;
    ui.screenKey.hidden = !key;
    ui.screenAction.hidden = !(phase === "complete" && s.mode === "human");   // B: the robot repeats it
    const thumb = board ? thumbSVG(board, side()) : "";
    if (ui.screenThumb.innerHTML !== thumb) ui.screenThumb.innerHTML = thumb;
    ui.screenThumb.hidden = !board;
  }

  // ------------------------------------------------------------ measured block colors

  // perception.calibrate writes each block's measured color as hex; use it when present so the
  // monitor shows the blocks on the table, not a generic palette. Missing file: CSS defaults.
  function applyBlockColors() {
    for (const [color, hex] of Object.entries(app.blockHex)) {
      if (COLORS.includes(color)) document.documentElement.style.setProperty(`--cube-${color}`, hex);
    }
    if (app.view3d) guard(app.view3d.setColors, app.blockHex);
  }
  fetch("/perception/calib/colors.json", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      const colors = (d && d.colors) || {};
      for (const [color, ref] of Object.entries(colors)) {
        if (ref && /^#[0-9a-f]{6}$/i.test(ref.hex || "")) app.blockHex[color] = ref.hex;
      }
      applyBlockColors();
    })
    .catch(() => {});

  render();
  connect();
  setView(app.view);
})();
