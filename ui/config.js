// Monitor settings. The student's side is not set here: it comes from the brain in every
// SessionState (human_side, brain/settings.py HUMAN_SIDE).
// Override the brain address with ?ws=ws://host:8766 and the view with ?view=2d or ?view=3d.
window.APPRENTICE_CONFIG = {
  BRAIN_WS_PORT: 8766,
  PASS_HOLD_MS: 1600,   // keep the green check up this long, even if the next step arrives at once
  RECONNECT_MIN_MS: 500,
  RECONNECT_MAX_MS: 3000,
};
