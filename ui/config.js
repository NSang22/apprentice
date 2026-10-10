// Monitor settings. HUMAN_SIDE must match brain/settings.py (tests/test_ui_config.py checks it).
// "row3": the student stands opposite the arm, column A on their left, row 3 nearest them.
// Override the brain address with ?ws=ws://host:8766 and the view with ?view=3d.
window.APPRENTICE_CONFIG = {
  HUMAN_SIDE: "row3",
  BRAIN_WS_PORT: 8766,
  PASS_HOLD_MS: 1600,   // keep the green check up this long, even if the next step arrives at once
  RECONNECT_MIN_MS: 500,
  RECONNECT_MAX_MS: 3000,
};
