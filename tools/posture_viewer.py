#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "numpy", "opencv-python"]
# ///
"""
Posture readout for the 54x42 mask, read through the RP2040 bridge.

    uv run tools/posture_viewer.py COM8

viewer.py shows the layers; this one says what the person is doing with them.
It consumes TYPE_MASK - the layer the RP2040 has already thresholded, opened,
hole-filled and reduced to the largest blob - and reports a posture label plus
the two scalars posture.py produces for a state machine downstream.

The serial framing, the CRC and the port handling all come from viewer.py; only
the judgement and the panel are new here.

Calibration is one step: sit the way you want everything measured against and
press SPACE. Every number after that is relative to that posture, which is what
lets the same thresholds work for different people and sitting distances.

Keys:
    SPACE   capture the reference posture      q / ESC   quit
    r       recapture it                       s         save a snapshot PNG
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer
from posture import W, H, Tracker, Verdict

LABEL_COLOUR = {          # BGR
    "UPRIGHT": (110, 220, 110),
    "SLUMP": (70, 70, 245),
    "RECLINE": (60, 210, 245),
    "DROWSY": (200, 120, 255),
    "ABSENT": (150, 150, 150),
    "BASELINE": (240, 200, 90),
    "UNKNOWN": (180, 180, 180),
}
PANEL_W = 320
PANEL_MIN_H = 560
STALE_S = 1.0             # no mask for this long and the readout is not live


def draw_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[mask.astype(bool)] = (235, 235, 235)
    return cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)


def _text(img, s, org, size=0.44, colour=(235, 235, 235), weight=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, weight, cv2.LINE_AA)


def _bar(img, org, width, value, colour):
    x, y = org
    cv2.rectangle(img, (x, y), (x + width, y + 10), (55, 55, 60), -1)
    filled = int(width * float(np.clip(value, 0.0, 1.0)))
    if filled:
        cv2.rectangle(img, (x, y), (x + filled, y + 10), colour, -1)
    cv2.rectangle(img, (x, y), (x + width, y + 10), (95, 95, 100), 1)


def panel(v: Verdict, height: int, *, ready: bool, fps: float, stale: bool) -> np.ndarray:
    p = np.full((height, PANEL_W, 3), 26, np.uint8)
    bar_w = PANEL_W - 32
    y = 30

    if not ready:
        _text(p, "sit as you want measured,", (16, y), 0.44, (240, 200, 90))
        _text(p, "then press SPACE", (16, y + 18), 0.44, (240, 200, 90))
        y += 46

    _text(p, "POSTURE", (16, y), 0.42, (150, 150, 155))
    _text(p, v.label, (16, y + 38), 1.0, LABEL_COLOUR.get(v.label, (200, 200, 200)), 2)
    if v.note:
        _text(p, v.note, (16, y + 58), 0.40, (165, 165, 170))
    y += 86

    for key, val, colour in (("phi   (focus)", v.phi, (200, 190, 110)),
                             ("delta (fatigue)", v.delta, (90, 130, 245))):
        _text(p, key, (16, y))
        _text(p, f"{val:.2f}", (PANEL_W - 56, y), 0.44, colour)
        _bar(p, (16, y + 6), bar_w, val, colour)
        y += 40

    y += 8
    _text(p, "deviation", (16, y), 0.42, (150, 150, 155))
    y += 22
    for key, colour in (("slump", (70, 70, 245)), ("recline", (60, 210, 245)),
                        ("drowsy", (200, 120, 255))):
        val = v.parts.get(key, 0.0)
        _text(p, key, (16, y))
        _text(p, f"{val:.2f}", (PANEL_W - 56, y), 0.44, colour)
        _bar(p, (16, y + 6), bar_w, val, colour)
        y += 38

    y += 10
    _text(p, "geometry", (16, y), 0.42, (150, 150, 155))
    f = v.features
    for key, val in (("occupancy", f"{f.occupancy * 100:.1f} %"),
                     ("head row", f"{f.top_row:.3f}"),
                     ("spread", f"{f.spread:.3f}"),
                     ("scale", f"{f.scale:.3f}"),
                     ("shrink", f"{v.parts.get('shrink', 0.0) * 100:+.1f} %"),
                     ("nod / min", f"{v.nod_rate:.1f}")):
        y += 20
        _text(p, key, (16, y), 0.42, (175, 175, 180))
        _text(p, val, (PANEL_W - 110, y), 0.42)

    _text(p, "NO MASK - is the link up?" if stale else f"{fps:4.1f} fps",
          (16, height - 32), 0.42, (70, 70, 245) if stale else (150, 150, 155))
    _text(p, "SPACE reference   r redo   s save   q quit", (16, height - 12), 0.38,
          (140, 140, 145))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="serial port, e.g. COM8 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600, help="must match LINK_BAUD")
    ap.add_argument("--scale", type=int, default=10, help="pixel magnification")
    ap.add_argument("--reference-frames", type=int, default=60,
                    help="frames averaged into the reference posture")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list or not args.port:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"{p.device}  {p.description}")
        if not args.port:
            print("\nno port given", file=sys.stderr)
            return 2
        return 0

    link = viewer.Link(args.port, args.baud)
    link.start()
    tracker = Tracker()
    print("waiting for TYPE_MASK frames. Sit as you want measured, then press SPACE.")

    fps, last = 0.0, time.perf_counter()
    blank = np.zeros((H, W), np.uint8)
    verdict = None
    try:
        while link.running:
            frames, _bad, _total = link.snapshot()
            mask = frames.get(viewer.TYPE_MASK)
            stamp = link.stamps.get(viewer.TYPE_MASK, 0.0)
            stale = mask is None or time.monotonic() - stamp > STALE_S

            now = time.perf_counter()
            verdict = tracker.update(blank if mask is None else mask, now)
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            img = draw_mask(blank if mask is None else mask, args.scale)
            height = max(img.shape[0], PANEL_MIN_H)
            if img.shape[0] < height:
                pad = height - img.shape[0]
                img = cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(18, 18, 20))
            canvas = np.hstack([img, panel(verdict, height, ready=tracker.ready,
                                           fps=fps, stale=stale)])
            cv2.imshow("posture", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("r")):
                tracker.start_baseline(args.reference_frames)
                print("capturing the reference posture - hold still")
            elif key == ord("s"):
                name = f"posture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, canvas)
                print(f"saved {name}")
    finally:
        link.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
