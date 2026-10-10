"""Brain-only settings. Team-wide contracts and thresholds live in shared/config.py.

These values only change brain behavior, so they live here and need no team sign-off.
"""

# Which side of the grid the human student stands on: "row1" (the README "operator side",
# columns A to C run left to right for them) or "row3" (across the board, so columns run
# right to left and row 3 is nearest them). Hint directions ("left", "toward you") follow it.
HUMAN_SIDE = "row3"
HUMAN_SIDES = ("row1", "row3")

# Hint ladder (human mode): 1 text, 2 ghost cube, 3 show-me clip, 4 robot hovers.
MAX_HINT_LEVEL = 4
SHOW_ME_PRE_S = 3.0   # clip starts this long before the cube appeared in the expert video
SHOW_ME_POST_S = 1.0  # and ends this long after
HOVER_HOLD_S = 3.0    # robot holds over the target cell this long, then returns home

# Robot HTTP client. Calls block until the motion finishes, so motion timeouts are generous.
ROBOT_MOTION_TIMEOUT_S = 30.0
ROBOT_STATUS_TIMEOUT_S = 2.0

# Fake robot and in-process sim feed (server --robot fake --grid sim).
FAKE_MOTION_S = 0.6   # how long each fake motion keeps the arm over the grid
SIM_FEED_HZ = 10.0
