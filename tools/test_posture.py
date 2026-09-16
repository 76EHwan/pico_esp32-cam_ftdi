#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Synthetic check for posture.py. No board and no camera needed.

    uv run tools/test_posture.py

Builds 54x42 masks by hand, drives them through the tracker and asserts the
label that comes out. Every threshold in posture.py is exercised by at least
one case, so this is what to run after changing one of them.

Exits non-zero on the first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posture import W, H, Tracker

FPS = 30.0
Y, X = np.mgrid[0:H, 0:W]


def person(head_row=6.0, scale=1.0, hand=None):
    """A head and a torso. scale < 1 is the same person further away."""
    cx = W / 2
    head_r = 5.0 * scale
    torso_rx, torso_ry = 13.0 * scale, 15.0 * scale
    head_cy = head_row + 4.0 * scale
    torso_cy = head_row + 20.0 * scale

    m = ((X - cx) ** 2 / head_r ** 2 + (Y - head_cy) ** 2 / head_r ** 2) <= 1.0
    m |= ((X - cx) ** 2 / torso_rx ** 2 + (Y - torso_cy) ** 2 / torso_ry ** 2) <= 1.0
    if hand:
        col = cx if hand == "over" else cx + 16
        m |= (np.abs(X - col) <= 1) & (Y <= head_row + 22)
    return m.astype(np.uint8)


def nod(t, period=6.0, rest=6.0, depth=9.0):
    """Head sinks over 1.5 s, snaps back over 0.5 s, then rests."""
    phase = t % period
    if phase < 1.5:
        return rest + depth * (phase / 1.5)
    if phase < 2.0:
        return rest + depth * (1.0 - (phase - 1.5) / 0.5)
    return rest


def calibrate(seconds=1.5):
    tr = Tracker()
    tr.start_baseline(int(seconds * FPS))
    t = 0.0
    for _ in range(int(seconds * FPS)):
        t += 1 / FPS
        tr.update(person(), t)
    assert tr.ready, "baseline was not captured"
    return tr, t


def hold(tr, t, frame_fn, seconds):
    """Run a pose for a while and return the last verdict."""
    n = int(seconds * FPS)
    for i in range(n):
        t += 1 / FPS
        v = tr.update(frame_fn(i / FPS), t)
    return v, t


CASES = [
    # name,                     pose,                                 seconds, expected
    ("upright",       lambda s: person(),                                 5.0, "UPRIGHT"),
    ("slump",         lambda s: person(head_row=19.0),                    5.0, "SLUMP"),
    ("recline",       lambda s: person(head_row=9.0, scale=0.78),         5.0, "RECLINE"),
    ("drowsy",        lambda s: person(head_row=nod(s)),                 40.0, "DROWSY"),
    ("absent",        lambda s: np.zeros((H, W), np.uint8),               2.0, "ABSENT"),
    ("hand to side",  lambda s: person(hand="side"),                      5.0, "UPRIGHT"),
    ("hand overhead", lambda s: person(hand="over"),                      5.0, "UPRIGHT"),
    # A nod must not be mistaken for a slump even though the head is just as low
    # at the bottom of it. Two seconds is longer than a dip, shorter than a hold.
    ("brief dip",     lambda s: person(head_row=19.0),                    2.0, "UPRIGHT"),
]

failed = 0
print(f"{'case':16s} {'got':9s} {'want':9s}  {'head':>5s} {'scale':>6s} "
      f"{'slump':>6s} {'recl':>5s} {'drow':>5s}")
for name, pose, seconds, want in CASES:
    tr, t = calibrate()
    v, t = hold(tr, t, pose, seconds)
    ok = v.label == want
    failed += not ok
    f, p = v.features, v.parts
    print(f"{name:16s} {v.label:9s} {want:9s}  {f.top_row:5.3f} {f.scale:6.3f} "
          f"{p.get('slump', 0):6.2f} {p.get('recline', 0):5.2f} {p.get('drowsy', 0):5.2f}"
          f"{'' if ok else '   <-- FAIL'}")

# Waking up has to clear quickly: the 30 s counting window would otherwise keep
# DROWSY on screen long after the nodding stopped.
tr, t = calibrate()
v, t = hold(tr, t, lambda s: person(head_row=nod(s)), 30.0)
assert v.label == "DROWSY", f"expected DROWSY before the recovery test, got {v.label}"
back = None
t1 = t
for i in range(int(40 * FPS)):
    t += 1 / FPS
    v = tr.update(person(), t)
    if back is None and v.label == "UPRIGHT":
        back = t - t1
print(f"\nwake-up recovery: {back:.1f} s" if back else "\nwake-up recovery: never")
if back is None or back > 20.0:
    failed += 1
    print("   <-- FAIL: should return to UPRIGHT within 20 s")

print(f"\n{'all passed' if not failed else f'{failed} failed'}")
sys.exit(1 if failed else 0)
