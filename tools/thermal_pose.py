#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "numpy", "mediapipe"]
# ///
# opencv-python is missing from that list on purpose, though the other tools
# here ask for it: mediapipe depends on opencv-contrib-python, which ships the
# same cv2 package, and naming both puts two distributions over the same files.
"""
ESP32-CAM -> pseudo-thermal filter -> human skeleton overlay.

    uv run tools/thermal_pose.py COM5

This is thermal-pose with the laptop webcam taken out and the ESP32-CAM put in
its place. Nothing downstream of the frame changed: the same pose model, the
same false-colour palettes, the same keys. The camera is the only difference.

Pipeline order matters:
    frame (upscaled preview) --> pose model --> landmarks + person mask
              |                                          |
              v                                          v
      thermal false-color  <--------------------  skeleton drawn on top

Running the pose model on the color-mapped image instead would tank detection
accuracy, so the filter is applied for display only.

What the swap actually costs:

* **160x120 greyscale, about 5 fps.** One preview frame is 19,212 bytes and the
  link runs at 921600 baud, so a frame costs ~208 ms of wire no matter how fast
  the sensor or the pose model is. The stream is the bottleneck here, not the
  CPU - the opposite of the webcam version, where pose inference was.
* **Every frame is upscaled before the pose model sees it** (`--scale`). The
  landmark model crops and resizes to its own input size, and handing it a
  160x120 image means that crop is mostly interpolation noise. Upscaling adds
  no information, but it does keep the detector working at a size it was
  trained around.
* **The sensor look is no longer a simulation.** The webcam version faked a
  low-resolution sensor by pixelating a 720p frame; here the source really is
  160x120, so `--cell` defaults to the upscale factor and the "sensor pixels"
  land exactly on the real ones. The grid starts off for the same reason -
  there is nothing left to imitate.

NOTE: this is not a thermal camera. The ESP32-CAM measures light, not
temperature. This is a false-color image (luminance mapped through a thermal
palette), and no pixel value means degrees.

Keys:
    q / ESC   quit                 m   cycle view (thermal / raw / split)
    c         cycle palette        k   toggle skeleton
    j         skeleton style (tee / upper body / limbs / all 33 landmarks)
    s         toggle body-heat segmentation
    z         render at the 54x42 ToF zone grid
    p         toggle low-res sensor simulation
    g         toggle sensor grid
    [ / ]     smaller / larger sensor pixels
    x         let the ESP32 re-expose (its AEC/AGC are locked at boot)
    space     save a snapshot
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    sys.exit("mediapipe is missing. Install it with: pip install mediapipe")

from palette import PALETTES, build_lut, sensor_grid, zone_grid  # noqa: F401
from viewer import Link, TYPE_PREVIEW

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

VIEWS = ("thermal", "raw", "split")

ALL_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

# Torso and limbs only. Real thermal footage has no facial detail to anchor to,
# so the 11 face landmarks just add clutter.
LIMB_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
LIMB_POINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# A camera sitting on the desk almost never sees hips or legs, so this set adds
# the head/neck line and drops everything below the waist.
UPPER_CONNECTIONS = [
    (0, 11), (0, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
UPPER_POINTS = {0, 11, 12, 13, 14, 15, 16, 23, 24}

# Head, shoulder bar, a stem rising from the middle of it, and an arm off each
# shoulder. CHEST is not a MediaPipe landmark - there is no chest in the model's
# 33 - so it is given an index past the end and filled in from the two shoulders
# at draw time.
NOSE, L_SHOULDER, R_SHOULDER, CHEST = 0, 11, 12, 33
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 13, 14, 15, 16
TEE_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER), (CHEST, NOSE),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
]
TEE_POINTS = {NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST}

# The grid the whole project targets: a VL53L9CX reports 2268 zones as 54x42.
ZONE_W, ZONE_H = 54, 42

STYLES = ("tee", "upper", "limbs", "all")
SKELETON = (255, 255, 255)
VIS_MIN = 0.5


# ---------------------------------------------------------------- the camera


class EspCamera:
    """The ESP32-CAM preview stream behind cv2.VideoCapture's read()/release().

    Everything above this class was written against a webcam and still is: it
    asks for a frame and gets a full-size BGR image back. The differences the
    link forces - greyscale, 160x120, one frame every ~208 ms - are absorbed
    here so that stays true.

    read() blocks until a frame arrives that the previous call did not already
    hand out. Returning the same array again would spend a whole pose inference
    re-deriving landmarks that cannot have moved, and VIDEO mode would count it
    as a second observation of a motionless subject.
    """

    def __init__(self, port, baud, scale, every=1, timeout=6.0):
        self.link = Link(port, baud)
        self.link.start()
        # p<n> is "send a full preview every n-th 54x42 frame". 1 is as fast as
        # the link goes; the 54x42 stream underneath it starves, which is fine
        # because nothing here reads it.
        self.link.send(f"p{max(every, 1)}")
        self.scale = scale
        self.timeout = timeout
        self.key = 255          # keypress seen while blocked, for the main loop
        self.raw = None         # last 160x120 frame, for the exposure readout
        self._last = None

    def read(self):
        deadline = time.monotonic() + self.timeout
        while self.link.running:
            img = self.link.snapshot()[0].get(TYPE_PREVIEW)
            if img is not None and img is not self._last:
                self._last = self.raw = img
                big = cv2.resize(img, (img.shape[1] * self.scale, img.shape[0] * self.scale),
                                 interpolation=cv2.INTER_CUBIC)
                return True, cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

            # Pump HighGUI while waiting. Four fifths of the wall clock is spent
            # in here, so without this the window stops repainting and a key
            # pressed between frames is simply lost.
            k = cv2.waitKey(10) & 0xFF
            if k != 255:
                self.key = k
            if time.monotonic() > deadline:
                return False, None
        return False, None

    def take_key(self):
        k, self.key = self.key, 255
        return k

    def send(self, line):
        self.link.send(line)

    def release(self):
        self.link.send("p0")
        time.sleep(0.1)         # let it go out before the port closes
        self.link.stop()


# ---------------------------------------------------------------- the filter


def work_size(w, h, pixelate, cell, zones=False):
    """Resolution the thermal image is actually computed at.

    The output is a low-resolution sensor image either way, so computing it at
    full frame size and then shrinking it is pure waste.

    `zones` pins it to the grid this whole project is aimed at - the 2268 cells
    a VL53L9CX reports - so the view stops being a stand-in for a low-resolution
    sensor and becomes the resolution the rest of the pipeline actually works
    at. The cells are very slightly wider than the sketch's, which crops 154 of
    the 160 columns before boxing down; matching that exactly would need the
    landmark coordinates remapped onto a cropped canvas for a 4% difference.
    """
    if zones:
        return ZONE_W, ZONE_H, cv2.INTER_NEAREST
    if pixelate:
        c = max(cell, 2)
        return max(w // c, 8), max(h // c, 6), cv2.INTER_NEAREST
    scale = min(1.0, 480.0 / max(h, 1))
    return max(int(w * scale), 8), max(int(h * scale), 6), cv2.INTER_LINEAR


def to_heat(frame, mask, clahe, use_seg, sw, sh):
    """Map a BGR frame to a small single-channel 'temperature' image in [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (gray.shape[1], gray.shape[0]) != (sw, sh):
        gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    heat = clahe.apply(gray).astype(np.float32) / 255.0

    if use_seg and mask is not None:
        # Real thermal cameras show a warm body against a cool background. Raw
        # luminance does the opposite for e.g. a white wall, so lift the
        # person's band and push the background down.
        m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
        m = np.clip(cv2.GaussianBlur(m, (0, 0), max(sw / 320.0, 0.6)), 0.0, 1.0)
        heat = m * (0.55 + 0.45 * heat) + (1.0 - m) * (0.05 + 0.32 * heat)

    return np.clip(heat, 0.0, 1.0, out=heat)


def colorize(heat, palette):
    u8 = (heat * 255.0).astype(np.uint8)
    name, cmap = PALETTES[palette]
    if cmap is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), name
    return cv2.applyColorMap(u8, cmap), name


def render_thermal(frame, mask, clahe, use_seg, pixelate, cell, palette, grid, zones=False):
    """Full display path: frame + person mask -> full-size thermal BGR image."""
    h, w = frame.shape[:2]
    sw, sh, interp = work_size(w, h, pixelate, cell, zones)
    heat = to_heat(frame, mask, clahe, use_seg, sw, sh)
    small, name = colorize(heat, palette)
    out = cv2.resize(small, (w, h), interpolation=interp)
    if grid:
        # One line per cell boundary. The webcam version drew two, which was
        # invisible at 16 screen pixels a cell and would swallow the picture
        # whole at the 4-6 this camera works out to.
        if zones:
            zone_grid(out, sw, sh)
        else:
            sensor_grid(out, cell if pixelate else 8)
    return out, name


def draw_skeleton(canvas, poses, style="upper", thickness=None, radius=None):
    """Draw landmarks manually so the colors stay readable on any palette."""
    h, w = canvas.shape[:2]
    scale = max(h / 720.0, 0.5)
    thickness = thickness or max(int(8 * scale), 2)
    radius = radius or max(int(10 * scale), 3)
    if style == "all":
        connections, keep = ALL_CONNECTIONS, None
    elif style == "limbs":
        connections, keep = LIMB_CONNECTIONS, LIMB_POINTS
    elif style == "tee":
        connections, keep = TEE_CONNECTIONS, TEE_POINTS
    else:
        connections, keep = UPPER_CONNECTIONS, UPPER_POINTS

    total = 0
    for landmarks in poses:
        pts = {}
        for idx, lm in enumerate(landmarks):
            if lm.visibility < VIS_MIN or (keep is not None and idx not in keep):
                continue
            pts[idx] = (int(lm.x * w), int(lm.y * h))

        # The model has no chest landmark, so make one: the shoulders' midpoint.
        # Joining the head to that instead of to each shoulder separately turns
        # the triangle into a bar with a stem, which is what an upper body
        # actually looks like from the front - the head sits above the middle of
        # the shoulders, not at the apex of a tent.
        if style == "tee" and L_SHOULDER in pts and R_SHOULDER in pts:
            lx, ly = pts[L_SHOULDER]
            rx, ry = pts[R_SHOULDER]
            pts[CHEST] = ((lx + rx) // 2, (ly + ry) // 2)

        for a, b in connections:
            if a in pts and b in pts:
                cv2.line(canvas, pts[a], pts[b], SKELETON, thickness, cv2.LINE_AA)

        for p in pts.values():
            cv2.circle(canvas, p, radius, SKELETON, -1, cv2.LINE_AA)
        total += len(pts)
    return total


def merge_masks(masks):
    """Combine per-person masks into one float32 mask (left at model resolution)."""
    if not masks:
        return None
    out = None
    for m in masks:
        arr = m.numpy_view().astype(np.float32)
        out = arr if out is None else np.maximum(out, arr)
    return out


def hud(canvas, lines):
    y = 24
    for text in lines:
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("port", nargs="?", help="the Vision Stream port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=921600, help="must match LINK_BAUD")
    ap.add_argument("--scale", type=int, default=5,
                    help="upscale factor applied to the 160x120 preview (default 5)")
    ap.add_argument("--every", type=int, default=1,
                    help="preview every Nth camera frame (default 1, as fast as the link goes)")
    ap.add_argument("--model", default=None,
                    help="path to a .task pose landmarker model "
                         "(default: pose_landmarker_full.task next to this file)")
    ap.add_argument("--people", type=int, default=1, help="max number of people to track")
    ap.add_argument("--cell", type=int, default=0,
                    help="sensor pixel size in screen pixels (default: --scale, "
                         "which lands on the camera's own pixels)")
    ap.add_argument("--style", default="tee", choices=STYLES, help="skeleton landmark set")
    ap.add_argument("--no-zones", action="store_true",
                    help=f"start at the camera's own resolution instead of the "
                         f"{ZONE_W}x{ZONE_H} ToF zone grid")
    ap.add_argument("--sharp", action="store_true", help="start without the low-res sensor look")
    ap.add_argument("--grid", action="store_true", help="start with the sensor grid drawn")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--outdir", default="snapshots")
    args = ap.parse_args()

    if not args.port:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"{p.device:10s} {p.description}")
        print("\nGive it the Vision Stream port. 'uv run tools/viewer.py --list' labels them.")
        return 1

    model = Path(args.model) if args.model else Path(__file__).with_name("pose_landmarker_full.task")
    if not model.is_file():
        sys.exit(f"Pose model not found: {model}\nDownload one of:\n  "
                 + "\n  ".join(MODEL_URLS.values()))

    scale = max(args.scale, 1)
    cam = EspCamera(args.port, args.baud, scale, args.every)
    print(f"[open] {args.port} @ {args.baud} - waiting for the first preview frame\n")

    outdir = Path(args.outdir)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=args.people,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    view, palette = 0, 0
    style = STYLES.index(args.style)
    show_skeleton, use_seg = True, True
    pixelate = not args.sharp
    zones = not args.no_zones
    # The webcam version tied the grid to the sensor simulation. Here the source
    # really is 160x120, so the pixels the grid would outline are the sensor's
    # own and drawing over them only costs contrast. Still on the 'g' key.
    grid = args.grid
    cell = max(args.cell or scale, 2)
    auto_exposure = False       # the sketch locks AEC/AGC once at boot
    fps, last = 0.0, time.perf_counter()
    t0 = time.perf_counter()

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cam.read()
                if not ok:
                    print("\nNo camera frames. Is this the Vision Stream port, and is the "
                          "ESP32 running esp32cam_sender?")
                    break
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                # Pose runs on the untouched frame -- this is the whole point.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # VIDEO mode needs monotonically increasing timestamps to track.
                result = landmarker.detect_for_video(
                    mp_image, int((time.perf_counter() - t0) * 1000))

                mask = merge_masks(result.segmentation_masks)
                thermal, palette_name = render_thermal(
                    frame, mask, clahe, use_seg, pixelate, cell, palette, grid, zones)

                joints = 0
                if show_skeleton and result.pose_landmarks:
                    joints = draw_skeleton(thermal, result.pose_landmarks, STYLES[style])

                mode = VIEWS[view]
                if mode == "thermal":
                    canvas = thermal
                elif mode == "raw":
                    canvas = frame.copy()
                    if show_skeleton and result.pose_landmarks:
                        draw_skeleton(canvas, result.pose_landmarks, STYLES[style])
                else:
                    canvas = np.hstack([frame, thermal])

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now

                # spread is the number that says whether the picture is usable
                # at all: a covered lens and a blown-out frame are both flat,
                # and both look like a broken model from up here.
                raw = cam.raw
                lo, hi, mean = int(raw.min()), int(raw.max()), int(raw.mean())

                hud(canvas, [
                    f"{fps:4.1f} fps | view:{mode} | palette:{palette_name}",
                    f"skeleton:{STYLES[style]} ({joints} pts) | bodyheat:{'on' if use_seg else 'off'} "
                    f"| sensor:{f'{ZONE_W}x{ZONE_H} zones' if zones else (f'{cell}px' if pixelate else 'off')}"
                    f" | grid:{'on' if grid else 'off'}",
                    f"camera 160x120  min {lo:3d}  mean {mean:3d}  max {hi:3d}  "
                    f"spread {hi - lo:3d}  | aec:{'auto' if auto_exposure else 'locked'}",
                    "q quit  m view  c palette  k/j skeleton  s bodyheat  z zones  p sensor  g grid  "
                    "[ ] size  x expose  SPACE save",
                ])
                cv2.imshow("thermal pose", canvas)

                # A key pressed during the ~200 ms wait for the next frame was
                # already taken off HighGUI's queue in read().
                key = cam.take_key()
                if key == 255:
                    key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord("m"):
                    view = (view + 1) % len(VIEWS)
                elif key == ord("c"):
                    palette = (palette + 1) % len(PALETTES)
                elif key == ord("k"):
                    show_skeleton = not show_skeleton
                elif key == ord("j"):
                    style = (style + 1) % len(STYLES)
                elif key == ord("s"):
                    use_seg = not use_seg
                elif key == ord("z"):
                    zones = not zones
                elif key == ord("p"):
                    pixelate = not pixelate
                elif key == ord("g"):
                    grid = not grid
                elif key == ord("["):
                    cell = max(cell - 2, 2)
                elif key == ord("]"):
                    cell = min(cell + 2, 80)
                elif key == ord("x"):
                    # The sketch exposes once at boot and freezes there, because
                    # background subtraction cannot survive a live AEC loop. No
                    # background model is in use here, so the loop is free to
                    # run - and a frozen exposure from an hour ago is the usual
                    # reason a person is a silhouette or a white smear.
                    auto_exposure = not auto_exposure
                    cam.send(f"x{1 if auto_exposure else 0}")
                elif key == ord(" "):
                    outdir.mkdir(parents=True, exist_ok=True)
                    path = outdir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(path), canvas)
                    print(f"saved {path}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
