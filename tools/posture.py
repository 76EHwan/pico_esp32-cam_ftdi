"""
Posture judgement on top of the 54x42 mask.

The RP2040 already turns the ESP32's coverage field into a cleaned binary mask
(threshold, open/close, hole fill, largest blob). This stage takes that mask and
answers "what is the person doing", as a label plus two scalars a state machine
downstream can consume.

    mask (54x42) -> geometry -> baseline-relative deviation -> label + phi/delta

Nothing here knows it came from a camera. Feed it a mask from any sensor that
produces this grid and it behaves the same.

The four things it separates:

    UPRIGHT   the reference posture
    SLUMP     head down and *stays* down
    RECLINE   leaning back: the silhouette shrinks
    DROWSY    head bobbing down and up repeatedly
    ABSENT    almost nothing in frame

Two ideas do most of the work.

**Head height alone cannot tell slumping from reclining.** Both drop the head
in the image - lean back in a chair and your head sinks in frame just as it
does when you fold onto the desk. The discriminator has to be apparent size:
leaning back moves you away from the camera and shrinks you, folding forward
does not. So head height supplies *how far* from the reference, and the change
in silhouette scale decides *which way*.

**Slumping and nodding differ in time, not in shape.** At the instant a nod
bottoms out the geometry is indistinguishable from a slump. Slumping is
defined as staying down, so it only counts after the head has been low
continuously for SLUMP_HOLD_S; a nod is over long before that.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

W, H = 54, 42

PRESENT_MIN_OCC = 0.04    # occupied fraction below this reads as nobody there

# Head finding. Taking the topmost occupied row means a raised hand becomes the
# head - it is higher than the head and the arm's distance then stands in for
# the head's. A head is several cells wide and sits over the torso; a forearm
# is one or two cells and can be anywhere.
HEAD_MIN_WIDTH = 4
HEAD_BAND_ROWS = 4
TORSO_HALF_SPAN = 0.22    # fraction of frame width either side of the body axis

SLUMP_SPAN = 0.22         # head this much lower (as a fraction of frame height) = full
SLUMP_HOLD_S = 3.0        # head must stay down this long before it is a slump
SLUMP_ARM_AT = 0.25       # deviation that starts the hold timer

# Reclining shrinks the silhouette. Expressed as a fraction of the reference
# scale rather than in pixels, so it does not need recalibrating when the
# camera, its lens or the sitting distance change.
RECLINE_SHRINK_SPAN = 0.18
SCALE_DEADBAND = 0.04     # below this the size change is noise, not a lean

# A nod is a dip and a recovery. Amplitude is relative to the person's own size
# so it survives a different body or a different distance.
NOD_WINDOW_S = 30.0
NOD_AMPLITUDE_REL = 0.10
NOD_AMPLITUDE_MIN = 0.04
NOD_MIN_DOWN_S = 0.30     # shorter than this is jitter
NOD_MAX_DOWN_S = 2.5      # longer than this is looking at something, not nodding
NOD_REFRACTORY_S = 0.8
NOD_RATE_FULL = 12.0      # nods per minute for a full drowsy contribution
# One dip early in a session must not read as a high rate: dividing a single
# event by a two-second observation gives 30/min. The denominator needs a floor.
NOD_MIN_SPAN_S = 12.0
# Drowsiness is a present-tense state. Waiting for events to age out of a 30 s
# window keeps DROWSY on the screen for half a minute after someone wakes up,
# so quiet time fades it out directly.
NOD_QUIET_S = 8.0
NOD_FADE_S = 6.0

MOTION_SPAN = 0.06

SLUMP_LABEL_AT = 0.70
RECLINE_LABEL_AT = 0.55
DROWSY_LABEL_AT = 0.50


@dataclass
class Features:
    occupancy: float      # occupied fraction of the grid
    top_row: float        # head row, 0 at the top of the frame, 1 at the bottom
    spread: float         # head to lowest occupied row
    scale: float          # torso width as a fraction of frame width
    motion: float         # cells that changed since the previous frame


@dataclass
class Verdict:
    label: str
    present: bool
    phi: float            # contribution to "focused", [0, 1]
    delta: float          # contribution to "fatigued", [0, 1]
    features: Features
    parts: dict[str, float] = field(default_factory=dict)
    nod_rate: float = 0.0
    note: str = ""


def find_head(mask: np.ndarray) -> tuple[int, int, int] | None:
    """Head row and the column band of the body axis, or None if nothing fits.

    Returns the highest row that is both wide enough to be a head and sitting
    over the body's axis, which is what keeps a raised arm from being mistaken
    for one.
    """
    counts = mask.sum(axis=1)
    if not counts.any():
        return None
    rows, cols = mask.shape

    # The body axis: the column centroid of the widest rows, i.e. the torso.
    body = np.flatnonzero(counts >= max(counts.max() * 0.5, 1.0))
    col_weight = mask[body].sum(axis=0).astype(np.float64)
    if col_weight.sum() <= 0:
        return None
    centre = float(col_weight @ np.arange(cols) / col_weight.sum())
    half = max(TORSO_HALF_SPAN * cols, HEAD_MIN_WIDTH)
    lo, hi = int(max(centre - half, 0)), int(min(centre + half, cols))

    band = mask[:, lo:hi].sum(axis=1)
    candidates = np.flatnonzero(band >= HEAD_MIN_WIDTH)
    if candidates.size == 0:
        return None
    return int(candidates[0]), lo, hi


def extract(mask: np.ndarray, prev: np.ndarray | None = None) -> tuple[Features, np.ndarray]:
    """Geometry from one mask. Returns the features and the mask as a bool array."""
    occ = mask.astype(bool)
    rows, cols = occ.shape
    occupancy = float(occ.sum()) / float(occ.size)

    head = find_head(occ)
    row_any = occ.any(axis=1)
    if head is not None and row_any.any():
        top, _lo, _hi = head
        bottom = int(np.flatnonzero(row_any)[-1])
        top_row = top / (rows - 1)
        spread = max(bottom - top, 0) / (rows - 1)
        # Scale = torso width. Shoulders keep their width whether you sit up or
        # fold forward, so what moves it is distance - which is the point.
        widths = occ.sum(axis=1)
        wide = widths[widths >= max(widths.max() * 0.5, 1)]
        scale = float(np.median(wide)) / cols
    else:
        top_row = spread = 1.0
        scale = 0.0

    motion = 0.0 if prev is None or prev.shape != occ.shape else \
        float(np.logical_xor(occ, prev).sum()) / float(occ.size)
    return Features(occupancy, top_row, spread, scale, motion), occ


@dataclass
class NodDetector:
    """Counts dip-and-recover cycles of the head and reports them per minute.

    The reference is the median over the window, so it follows a posture change.
    That is what stops a sustained slump from registering as an endless nod:
    once the head has been low for a while the median is low too.
    """
    window_s: float = NOD_WINDOW_S
    amplitude: float = NOD_AMPLITUDE_MIN
    _hist: deque = field(default_factory=deque, repr=False)
    _events: deque = field(default_factory=deque, repr=False)
    _down_since: float | None = None
    _last_event: float = -1e9
    _abandoned: bool = False

    def reset(self) -> None:
        self._hist.clear()
        self._events.clear()
        self._down_since = None
        self._abandoned = False

    def update(self, now: float, top_row: float, present: bool, scale: float = 1.0) -> float:
        # With nobody there top_row flips between 1.0 and whatever noise is left,
        # which counts as a stream of nods if it is not gated here.
        if not present:
            self.reset()
            return 0.0

        self.amplitude = max(NOD_AMPLITUDE_REL * scale, NOD_AMPLITUDE_MIN)
        self._hist.append((now, top_row))
        while self._hist and now - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        while self._events and now - self._events[0] > self.window_s:
            self._events.popleft()

        if len(self._hist) >= 8:
            rest = float(np.median([v for _, v in self._hist]))
            up = top_row < rest + self.amplitude * 0.3
            if self._abandoned:
                if up:
                    self._abandoned = False
            elif self._down_since is None:
                if top_row > rest + self.amplitude:
                    self._down_since = now
            elif up:
                held = now - self._down_since
                self._down_since = None
                if (NOD_MIN_DOWN_S <= held <= NOD_MAX_DOWN_S
                        and now - self._last_event >= NOD_REFRACTORY_S):
                    self._events.append(now)
                    self._last_event = now
            elif now - self._down_since > NOD_MAX_DOWN_S:
                # Held down rather than bobbing. Void it so the eventual lift
                # does not get counted as a very slow nod.
                self._down_since = None
                self._abandoned = True

        span = max(now - self._hist[0][0], 1.0) if self._hist else self.window_s
        rate = len(self._events) * 60.0 / max(min(span, self.window_s), NOD_MIN_SPAN_S)

        quiet = now - self._last_event
        fade = float(np.clip(1.0 - (quiet - NOD_QUIET_S) / NOD_FADE_S, 0.0, 1.0))
        return rate * fade


@dataclass
class Baseline:
    """The reference posture everything is measured against."""
    top_row: float
    spread: float
    scale: float
    samples: int = 0
    clipped: bool = False   # head was cut off by the top of the frame

    @classmethod
    def from_features(cls, feats: list[Features]) -> "Baseline":
        if not feats:
            raise ValueError("no frames to build a baseline from")
        med = lambda a: float(np.median([getattr(f, a) for f in feats]))
        top = med("top_row")
        # A head touching row 0 means the frame cut it off. Measuring against
        # that baseline reads every normal posture as "head has dropped".
        return cls(top, med("spread"), med("scale"), len(feats), clipped=top < 0.02)


def judge(feats: Features, base: Baseline | None, nod_rate: float,
          slump_held_s: float = SLUMP_HOLD_S) -> Verdict:
    """Label and scores for one frame."""
    if feats.occupancy < PRESENT_MIN_OCC:
        return Verdict("ABSENT", False, 0.0, 0.0, feats, nod_rate=nod_rate,
                       note="low occupancy")
    if base is None:
        return Verdict("UNKNOWN", True, 0.0, 0.0, feats, nod_rate=nod_rate,
                       note="no baseline")

    # How far from the reference, with no sense of direction.
    drop = float(np.clip((feats.top_row - base.top_row) / SLUMP_SPAN, 0.0, 1.0))
    collapse = float(np.clip((base.spread - feats.spread) / max(base.spread, 1e-6), 0.0, 1.0))
    magnitude = float(np.clip(0.8 * drop + 0.2 * collapse, 0.0, 1.0))

    # Which direction: a shrinking silhouette means moving away, i.e. leaning
    # back. This is the only axis that separates reclining from slumping.
    if base.scale > 1e-6:
        shrink = (base.scale - feats.scale) / base.scale
        recline = float(np.clip((shrink - SCALE_DEADBAND) / RECLINE_SHRINK_SPAN, 0.0, 1.0))
    else:
        shrink, recline = 0.0, 0.0

    # Suppress slumping while the silhouette says the person is moving away.
    slump_raw = magnitude * (1.0 - recline)
    slump = slump_raw * float(np.clip(slump_held_s / SLUMP_HOLD_S, 0.0, 1.0))
    drowsy = float(np.clip(nod_rate / NOD_RATE_FULL, 0.0, 1.0))

    parts = {"slump": slump, "recline": recline, "drowsy": drowsy,
             "shrink": shrink, "slump_raw": slump_raw}
    # Reclining is bad posture but not necessarily tiredness, so it weighs less.
    delta = float(np.clip(max(0.95 * slump, 0.85 * drowsy, 0.45 * recline), 0.0, 1.0))
    stability = 1.0 - float(np.clip(feats.motion / MOTION_SPAN, 0.0, 1.0))
    phi = float(np.clip((1.0 - delta) * stability, 0.0, 1.0))

    # Reclining is tested first: with the order reversed, slump crosses its
    # threshold on the way back and RECLINE never appears. Slumping only
    # qualifies once the hold has elapsed, so a nod cannot reach it.
    if recline >= RECLINE_LABEL_AT:
        label = "RECLINE"
    elif slump >= SLUMP_LABEL_AT:
        label = "SLUMP"
    elif drowsy >= DROWSY_LABEL_AT:
        label = "DROWSY"
    else:
        label = "UPRIGHT"
    note = "baseline clipped - recapture" if base.clipped else ""
    return Verdict(label, True, phi, delta, feats, parts, nod_rate, note)


@dataclass
class Tracker:
    """Feed masks in, get verdicts out. Owns the baseline and the nod history."""
    baseline: Baseline | None = None
    nods: NodDetector = field(default_factory=NodDetector)
    _prev: np.ndarray | None = field(default=None, repr=False)
    _buf: list[Features] = field(default_factory=list, repr=False)
    _want: int = 0
    _slump_since: float | None = field(default=None, repr=False)

    def start_baseline(self, samples: int = 60) -> None:
        """Call with the person sitting in the posture to measure against."""
        self._buf, self._want = [], samples

    @property
    def ready(self) -> bool:
        return self.baseline is not None and self._want == 0

    def update(self, mask: np.ndarray, now: float) -> Verdict:
        feats, self._prev = extract(mask, self._prev)
        present = feats.occupancy >= PRESENT_MIN_OCC
        scale_ref = self.baseline.spread if self.baseline else feats.spread
        nod_rate = self.nods.update(now, feats.top_row, present, scale_ref)

        if self._want > 0:
            want = self._want
            if present:
                self._buf.append(feats)
            done = len(self._buf)
            if done >= want:
                self.baseline = Baseline.from_features(self._buf)
                self._want = 0
            return Verdict("BASELINE", present, 0.0, 0.0, feats,
                           nod_rate=nod_rate, note=f"baseline {done}/{want}")

        held = (now - self._slump_since) if self._slump_since is not None else 0.0
        verdict = judge(feats, self.baseline, nod_rate, held)

        if verdict.parts.get("slump_raw", 0.0) >= SLUMP_ARM_AT:
            if self._slump_since is None:
                self._slump_since = now
        else:
            self._slump_since = None
        return verdict
