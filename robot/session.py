"""Hands-free start and finish for scripts that move the arm.

    with ArmSession(arm, skills) as s:      # countdown, torque on, rise and park
        skills.pick(...); skills.place(...)
                                            # on exit: set down at the rest spot, torque off

No ENTER presses in the normal flow. Ctrl+C stops the arm where it is (the servos hold their
last command) and then asks what to do, because an emergency is the one time a person decides.
"""

import time

from .skills import SkillError

COUNTDOWN_S = 3


class ArmSession:
    def __init__(self, arm, skills, countdown_s=COUNTDOWN_S, log=print):
        self.arm, self.skills, self.countdown_s, self.log = arm, skills, countdown_s, log

    def __enter__(self):
        self.arm.connect()
        self.log(f"The arm will start moving in {self.countdown_s} s. Ctrl+C stops it; the 12V plug is the hard stop.")
        for i in range(self.countdown_s, 0, -1):
            self.log(f"  {i}...")
            time.sleep(1)
        self.arm.enable()  # holds wherever the arm is
        self.skills.wake()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.arm.torque_on:
            self.arm.disconnect()
            return False
        if exc_type is KeyboardInterrupt:
            self.arm.hold()
            self._ask_after_stop()
            return True  # handled
        try:
            self.skills.rest()
        except (SkillError, Exception) as e:  # can't reach the rest spot: don't drop the arm silently
            self.log(f"Couldn't set the arm down at its rest spot ({e}).")
            self._ask_after_stop()
            return exc_type is None
        self.arm.disconnect()
        self.log("Arm at rest, torque off.")
        return False  # let a SkillError from the run propagate to the caller's handler

    def _ask_after_stop(self):
        self.log("\nArm stopped and holding.")
        try:
            ans = input("ENTER = set it down at the rest spot and release.  Ctrl+C = release torque NOW "
                        "(support the arm first): ")
        except (KeyboardInterrupt, EOFError):
            ans = "now"
        if ans != "now":
            try:
                self.skills.rest()
            except Exception as e:
                self.log(f"Rest failed ({e}); releasing torque, support the arm.")
                time.sleep(2)
        self.arm.disconnect()
        self.log("Torque off.")
