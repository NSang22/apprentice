// Shared top bar for the whole site: the hub (/app), the monitor (/ui/), and every Buddy page.
// Served by jetson-kushi/dashboard.py at /app/nav.js and included with one line:
//   <script src="/app/nav.js" defer></script>
// It only adds a bar (and its own styles); it never changes a page's logic. If it fails to load,
// every page works exactly as before.
(function () {
  "use strict";
  if (window.__apxNav) return;
  window.__apxNav = true;

  const NAV_H = 56;
  const MONITOR_H = 48;
  const MONITOR_IDLE_MS = 3000;
  const path = location.pathname;
  const isHub = path === "/app" || path.startsWith("/app/");
  const isMonitor = path === "/ui" || path.startsWith("/ui/");
  const isBuddy = path === "/buddy" || path.startsWith("/buddy/");
  // Buddy screens that are not a round: leaving from these needs no question.
  const BUDDY_SAFE_SCREENS = ["s-hello", "s-home", "s-progress", "s-over", "s-intro"];

  const TABS = [
    { key: "teach", label: "Teach", href: "/app#teach" },
    { key: "learn", label: "Learn", href: "/app#lessons" },
    { key: "play", label: "Play", href: "/buddy" },
    { key: "track", label: "Track", href: "/app#track" },
  ];

  // Pip's head, from buddy/static/index.html (#pip-head), with fixed colors so no page CSS is needed.
  const PIP = '<svg viewBox="16 4 88 90" width="40" height="40" aria-hidden="true">' +
    '<line x1="60" y1="22" x2="60" y2="9" stroke="#0e9a84" stroke-width="4" stroke-linecap="round"/>' +
    '<circle cx="60" cy="8" r="6" fill="#ffc800"/>' +
    '<rect x="16" y="26" width="88" height="68" rx="28" fill="#0e9a84"/>' +
    '<rect x="16" y="22" width="88" height="68" rx="28" fill="#14c4a8"/>' +
    '<rect x="27" y="34" width="66" height="44" rx="19" fill="#fff"/>' +
    '<ellipse cx="47" cy="54" rx="7" ry="9" fill="#2b2b2b"/><ellipse cx="73" cy="54" rx="7" ry="9" fill="#2b2b2b"/>' +
    '<path d="M51 66 Q60 74 69 66" fill="none" stroke="#2b2b2b" stroke-width="3.2" stroke-linecap="round"/></svg>';
  const BACK = '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M15 5l-7 7 7 7" fill="none" ' +
    'stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const RIG = "/";   // the rig dashboard (jetson-kushi/dashboard.py serves it at /; there is no /rig)
  const MENU = '<svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" ' +
    'fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>';

  const CSS = `
#apx-nav, #apx-nav * { box-sizing: border-box; }
#apx-nav { position: fixed; top: 0; left: 0; right: 0; height: ${NAV_H}px; z-index: 2147483000; display: flex; align-items: center;
  gap: 8px; padding: 0 12px; background: #fff; border-bottom: 2px solid #e5e5e5; color: #3c3c3c; cursor: auto;
  font: 800 15px/1 "Nunito", ui-rounded, "SF Pro Rounded", system-ui, sans-serif; -webkit-user-select: none; user-select: none; }
#apx-nav a, #apx-nav button { color: inherit; text-decoration: none; font: inherit; background: none; border: 0; cursor: pointer;
  -webkit-tap-highlight-color: transparent; }
#apx-nav .apx-home { display: flex; align-items: center; gap: 8px; min-height: 48px; padding: 0 8px 0 2px; border-radius: 16px;
  font-size: 20px; font-weight: 900; }
#apx-nav .apx-home svg { flex: none; }
#apx-nav .apx-tabs { display: flex; gap: 6px; margin-left: 12px; flex: 1; }
#apx-nav .apx-tab { display: flex; align-items: center; min-height: 48px; padding: 0 16px; border-radius: 16px; border: 2px solid transparent;
  letter-spacing: .06em; text-transform: uppercase; color: #777; }
#apx-nav .apx-tab:hover { background: #f7f7f7; }
#apx-nav .apx-tab.on { color: #1cb0f6; border-color: #84d8ff; background: #ddf4ff; box-shadow: 0 3px 0 #84d8ff; }
#apx-nav .apx-tab:active, #apx-nav .apx-icon:active { transform: translateY(3px); box-shadow: none; }
#apx-nav .apx-right { display: flex; align-items: center; gap: 6px; margin-left: auto; }
#apx-nav .apx-icon { width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 16px;
  color: #afafaf; }
#apx-nav .apx-icon:hover { background: #f7f7f7; color: #777; }
#apx-nav .apx-grown { min-height: 48px; display: flex; align-items: center; padding: 0 10px; font-size: 13px; font-weight: 800;
  color: #afafaf; }
#apx-nav .apx-menu { display: none; }
#apx-nav [hidden] { display: none !important; }
#apx-drop { position: fixed; top: ${NAV_H}px; left: 0; right: 0; z-index: 2147483000; background: #fff; border-bottom: 2px solid #e5e5e5;
  padding: 8px 12px 12px; display: none; flex-direction: column; gap: 6px; font: 800 16px/1 "Nunito", ui-rounded, system-ui, sans-serif; }
#apx-drop.open { display: flex; }
#apx-drop a { min-height: 48px; display: flex; align-items: center; padding: 0 16px; border-radius: 16px; color: #777; text-decoration: none;
  letter-spacing: .06em; text-transform: uppercase; }
#apx-drop a.on { color: #1cb0f6; background: #ddf4ff; }
#apx-drop a.apx-grown { text-transform: none; letter-spacing: 0; font-size: 14px; color: #afafaf; }
@media (max-width: 599px) {
  #apx-nav .apx-tabs, #apx-nav .apx-word, #apx-nav .apx-grown { display: none; }
  #apx-nav .apx-menu { display: flex; }
}

/* The monitor: slim and dark over the 3D view; it hides after a few still seconds. */
#apx-nav.apx-dark { height: ${MONITOR_H}px; background: rgba(13, 16, 21, .72); border-bottom-color: rgba(255,255,255,.08); color: #f3f5f8;
  -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); transition: transform .3s, opacity .3s; }
#apx-nav.apx-dark .apx-home { font-size: 17px; }
#apx-nav.apx-dark .apx-home svg { width: 32px; height: 32px; }
#apx-nav.apx-dark .apx-tab { color: #95a1b2; }
#apx-nav.apx-dark .apx-tab:hover, #apx-nav.apx-dark .apx-icon:hover { background: rgba(255,255,255,.08); }
#apx-nav.apx-dark .apx-tab.on { color: #f3f5f8; background: rgba(124,183,255,.18); border-color: rgba(124,183,255,.4); box-shadow: none; }
#apx-nav.apx-dark .apx-icon { color: #95a1b2; }
#apx-nav.apx-dark.apx-away { transform: translateY(-100%); opacity: 0; pointer-events: none; }
#apx-drop.apx-dark { top: ${MONITOR_H}px; background: #161b23; border-bottom-color: #2d3644; }
#apx-drop.apx-dark a { color: #95a1b2; }

/* Buddy: "Leave this game?" */
#apx-ask { position: fixed; inset: 0; z-index: 2147483001; background: rgba(0,0,0,.35); display: flex; align-items: flex-end;
  justify-content: center; font: 700 17px/1.3 "Nunito", ui-rounded, system-ui, sans-serif; color: #3c3c3c; }
#apx-ask .apx-sheet { width: 100%; max-width: 480px; background: #fff; border-radius: 22px 22px 0 0; padding: 22px 16px calc(18px + env(safe-area-inset-bottom)); }
#apx-ask h3 { margin: 0 0 16px; font-size: 22px; font-weight: 900; }
#apx-ask button { display: flex; align-items: center; justify-content: center; width: 100%; min-height: 52px; border: 0; border-radius: 16px;
  font: 800 17px "Nunito", ui-rounded, system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase; cursor: pointer; }
#apx-ask button:active { transform: translateY(4px); box-shadow: none; }
#apx-ask .apx-stay { background: #58cc02; color: #fff; box-shadow: 0 4px 0 #58a700; }
#apx-ask .apx-go { margin-top: 12px; background: #fff; color: #1cb0f6; border: 2px solid #e5e5e5; box-shadow: 0 4px 0 #e5e5e5; }

/* Pages that pin things to the top edge: move them below the bar. */
html { scroll-padding-top: ${NAV_H + 8}px; }
body.apx-has-nav .toast { top: ${NAV_H + 14}px; }
body.apx-has-nav.has3d.ingame #board3d, body.apx-has-nav.has3d.ingame #s-game { top: ${NAV_H}px; }
body.apx-has-nav.has3d.ingame .gridcard { top: ${NAV_H + 16}px; }
body.apx-has-nav .tv-top { top: ${NAV_H}px; }
`;

  function current() {
    if (isBuddy) return "play";
    if (isMonitor) return "learn";
    if (isHub) {
      const h = location.hash;
      if (h === "#track") return "track";
      if (h === "#lessons" || h === "#library") return "learn";
      if (h === "#teach") return "teach";
    }
    return null;
  }

  function inBuddyRound() {
    if (!isBuddy) return false;
    const on = document.querySelector(".screen.on");
    return !!on && !BUDDY_SAFE_SCREENS.includes(on.id);
  }

  // During a Buddy round, ask before leaving, so a child cannot quit by accident.
  function guard(ev, go) {
    if (!inBuddyRound()) return;
    ev.preventDefault();
    if (document.getElementById("apx-ask")) return;
    const ask = document.createElement("div");
    ask.id = "apx-ask";
    ask.innerHTML = '<div class="apx-sheet" role="dialog" aria-modal="true" aria-label="Leave this game?">' +
      '<h3>Leave this game?</h3><button class="apx-stay">Keep playing</button><button class="apx-go">Yes</button></div>';
    const close = () => ask.remove();
    ask.addEventListener("click", (e) => { if (e.target === ask) close(); });
    ask.querySelector(".apx-stay").onclick = close;
    ask.querySelector(".apx-go").onclick = () => { close(); go(); };
    document.body.appendChild(ask);
  }

  function build() {
    const style = document.createElement("style");
    style.id = "apx-nav-style";
    style.textContent = CSS;
    document.head.appendChild(style);

    const nav = document.createElement("nav");
    nav.id = "apx-nav";
    nav.setAttribute("aria-label", "Apprentice");
    const tabs = TABS.map((t) => `<a class="apx-tab" data-key="${t.key}" href="${t.href}">${t.label}</a>`).join("");
    nav.innerHTML =
      `<a class="apx-home" href="/app" title="Home" aria-label="Home">${PIP}<span class="apx-word">Apprentice</span></a>` +
      `<div class="apx-tabs">${tabs}</div>` +
      '<div class="apx-right">' +
      `<a class="apx-icon apx-back" href="#" title="Back" aria-label="Back" role="button">${BACK}</a>` +
      (isHub ? `<a class="apx-grown" href="${RIG}">Grown-ups</a>` : "") +
      `<button class="apx-icon apx-menu" type="button" aria-label="Menu" aria-expanded="false">${MENU}</button>` +
      "</div>";

    const drop = document.createElement("div");
    drop.id = "apx-drop";
    drop.innerHTML = TABS.map((t) => `<a data-key="${t.key}" href="${t.href}">${t.label}</a>`).join("") +
      (isHub ? `<a class="apx-grown" href="${RIG}">Grown-ups</a>` : "");

    if (isMonitor) { nav.classList.add("apx-dark"); drop.classList.add("apx-dark"); }
    document.body.prepend(drop);
    document.body.prepend(nav);
    document.body.classList.add("apx-has-nav");
    // The monitor's bar floats over the board and hides itself; everywhere else it pushes the page down.
    if (!isMonitor) {
      const pad = parseFloat(getComputedStyle(document.body).paddingTop) || 0;
      document.body.style.paddingTop = `${pad + NAV_H}px`;
    }

    const back = nav.querySelector(".apx-back");
    back.hidden = history.length <= 1;
    back.addEventListener("click", (ev) => {
      ev.preventDefault();
      guard(ev, () => history.back());
      if (!inBuddyRound()) history.back();
    });

    // Any link in the bar: ask first during a Buddy round; then leave.
    for (const a of [...nav.querySelectorAll("a[href]:not(.apx-back)"), ...drop.querySelectorAll("a[href]")]) {
      a.addEventListener("click", (ev) => {
        drop.classList.remove("open");
        guard(ev, () => { location.href = a.href; });
        a.blur();   // keep keys (Space on the monitor) going to the page
      });
    }

    const menu = nav.querySelector(".apx-menu");
    menu.addEventListener("click", () => {
      const open = drop.classList.toggle("open");
      menu.setAttribute("aria-expanded", String(open));
      menu.blur();
    });

    // The hub's own sections: the hub shows or hides its Track view on hashchange, so scroll after it has.
    if (isHub) {
      addEventListener("hashchange", () => requestAnimationFrame(() => {
        const id = location.hash.slice(1);
        const el = id && id !== "track" ? document.getElementById(id) : null;
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      }));
    }

    const mark = () => {
      const key = current();
      for (const a of document.querySelectorAll("#apx-nav [data-key], #apx-drop [data-key]")) {
        a.classList.toggle("on", a.dataset.key === key);
        if (a.dataset.key === key) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
      }
    };
    mark();
    addEventListener("hashchange", mark);

    if (isMonitor) {
      let timer = 0;
      const wake = () => {
        nav.classList.remove("apx-away");
        clearTimeout(timer);
        timer = setTimeout(() => {
          if (nav.matches(":hover") || drop.classList.contains("open")) return wake();
          nav.classList.add("apx-away");
        }, MONITOR_IDLE_MS);
      };
      for (const type of ["mousemove", "pointerdown", "touchstart"]) addEventListener(type, wake, { passive: true });
      wake();
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
