#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Synthetic check for posture_pose.py. No board, no camera, no mediapipe.

    uv run tools/test_posture_pose.py

Builds landmark sets by hand, feeds them through the tracker at a plausible
frame rate, and asserts the label. Every rule and every threshold in
posture_pose.py is exercised by at least one case here.
"""

import sys

from posture_pose import (ABSENT, CHIN_REST, RECLINE, SLUMP, UPRIGHT,
                          PostureTracker)

FPS = 5.0           # what the link actually delivers
DT = 1.0 / FPS


def figure(head_y=0.16, width=0.40, mid_y=0.46, wrist=None, elbow_down=True):
    """A front-facing upper body, in 0..1 image coordinates.

    head_y   nose height; larger is lower in frame
    width    shoulder separation, the apparent-size proxy
    mid_y    shoulder line height
    wrist    where to put the near wrist, or None to leave the arms down
    """
    half = width / 2.0
    pts = {
        0: (0.50, head_y),
        11: (0.50 - half, mid_y),
        12: (0.50 + half, mid_y),
        13: (0.50 - half - 0.06, mid_y + (0.18 if elbow_down else -0.18)),
        14: (0.50 + half + 0.06, mid_y + (0.18 if elbow_down else -0.18)),
    }
    if wrist is None:
        pts[15] = (0.50 - half - 0.02, mid_y + 0.36)
        pts[16] = (0.50 + half + 0.02, mid_y + 0.36)
    else:
        pts[15] = wrist
        pts[16] = (0.50 + half + 0.02, mid_y + 0.36)
    return pts


def run(tracker, pts, seconds, t0):
    """Hold a posture for a while and return the final state."""
    t = t0
    state = None
    for _ in range(max(int(seconds * FPS), 1)):
        state = tracker.update(pts, t)
        t += DT
    return state, t


UPRIGHT_POSE = figure()

# Leaning back: the head sinks AND the whole person gets smaller, because
# rocking back moves the shoulders away from the camera.
RECLINE_POSE = figure(head_y=0.34, width=0.33, mid_y=0.50)

# Folding onto the desk: the head sinks just as far, but the shoulders do not
# recede - this is the pair that head height alone cannot separate.
SLUMP_POSE = figure(head_y=0.36, width=0.41, mid_y=0.48)

# Chin on hand: the head barely moves, a wrist arrives beside it, elbow propped
# below the shoulder.
CHIN_POSE = figure(head_y=0.19, wrist=(0.42, 0.26))

# Same hand height, but the elbow is UP - waving, not resting.
WAVE_POSE = figure(head_y=0.16, wrist=(0.42, 0.24), elbow_down=False)

# Measured off a real camera sitting below desk height: lying back put the
# shoulders at 0.19 of the reference width and moved the head the *other* way,
# so any rule that also demanded a neck foreshortening vetoed it.
LYING_BACK_POSE = figure(head_y=0.30, width=0.076, mid_y=0.62)

CASES = [
    ("upright",        UPRIGHT_POSE, 3.0, UPRIGHT),
    ("recline",        RECLINE_POSE, 3.0, RECLINE),
    ("lying back",     LYING_BACK_POSE, 3.0, RECLINE),
    ("slump",          SLUMP_POSE,   3.0, SLUMP),
    ("chin rest",      CHIN_POSE,    3.0, CHIN_REST),
    ("wave, not chin", WAVE_POSE,    3.0, UPRIGHT),
    ("absent",         None,         3.0, ABSENT),
]


def main():
    fails = 0
    print(f"{'case':16s} {'got':10s} {'want':10s} {'drop':>6s} {'scale':>6s} "
          f"{'wrist':>6s}  note")

    for name, pose, secs, want in CASES:
        tr = PostureTracker()
        assert tr.capture_reference(UPRIGHT_POSE), "reference must take"
        state, _ = run(tr, pose, secs, 0.0)
        m = state.metrics
        ok = state.label == want
        fails += not ok
        wrist = "inf" if m.wrist_to_head > 9 else f"{m.wrist_to_head:.2f}"
        print(f"{name:16s} {state.label:10s} {want:10s} {m.head_drop_delta:6.2f} "
              f"{m.scale_ratio:6.2f} {wrist:>6s}  {state.note}"
              + ("" if ok else "   <-- FAIL"))

    # A nod is a slump that does not last. At the bottom the geometry is
    # identical, so only the hold timer can tell them apart.
    tr = PostureTracker()
    tr.capture_reference(UPRIGHT_POSE)
    t = 0.0
    _, t = run(tr, UPRIGHT_POSE, 3.0, t)
    state, t = run(tr, SLUMP_POSE, 0.6, t)       # a dip shorter than the hold
    if state.label != UPRIGHT:
        print(f"\nFAIL: a 0.6 s dip reported {state.label}, want {UPRIGHT}")
        fails += 1
    else:
        print(f"\nbrief dip (0.6 s) stayed {state.label}, candidate was {state.candidate}")

    # And it must still commit once the dip lasts.
    state, t = run(tr, SLUMP_POSE, 3.0, t)
    if state.label != SLUMP:
        print(f"FAIL: a sustained dip reported {state.label}, want {SLUMP}")
        fails += 1
    else:
        print(f"sustained dip became {state.label} after {tr.hold:.1f} s")

    # Without a reference nothing can be relative, so it must say so rather
    # than guess.
    tr2 = PostureTracker()
    state = tr2.update(SLUMP_POSE, 0.0)
    if state.has_reference or state.label != UPRIGHT:
        print(f"FAIL: no reference should report UPRIGHT/unreferenced, got "
              f"{state.label} has_reference={state.has_reference}")
        fails += 1
    else:
        print(f"no reference -> {state.label}, note: {state.note!r}")

    # One shoulder hidden: shoulder width is the ruler, so there is nothing to
    # measure against and it must not invent a posture.
    one = {k: v for k, v in UPRIGHT_POSE.items() if k != 12}
    tr3 = PostureTracker()
    tr3.capture_reference(UPRIGHT_POSE)
    state, _ = run(tr3, one, 3.0, 0.0)
    if state.label != ABSENT:
        print(f"FAIL: one shoulder reported {state.label}, want {ABSENT}")
        fails += 1
    else:
        print(f"one shoulder -> {state.label}, note: {state.note!r}")

    print("\n" + ("all passed" if not fails else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
