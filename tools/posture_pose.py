#!/usr/bin/env python3
"""
Posture from pose landmarks: upright, reclined, slumped, chin resting.

posture.py judges the same thing from the 54x42 mask, which is all a depth
sensor gives. This one takes named joints instead, so it can use the arms - and
the arms are what tell a chin rest from anything else.

Nothing here imports mediapipe or cv2. It takes seven points and a timestamp.

    tracker = PostureTracker()
    tracker.capture_reference(pts)          # sitting how you mean to sit
    state = tracker.update(pts, time.monotonic())
    state.label                             # UPRIGHT / RECLINE / SLUMP / CHIN_REST / ABSENT

`pts` maps landmark index -> (x, y) in any consistent units; image pixels or
0..1 both work, because every measurement below is divided by shoulder width.

WHY THE MEASUREMENTS ARE THE SHAPE THEY ARE

*Leaning back and folding forward both drop the head in the image.* Rocking
back tips the head away from the camera and it sinks in frame; folding onto the
desk drops it just as far. Height alone says how far from upright, never which
way. What separates them is apparent size: leaning back moves you away from the
camera and the shoulders narrow, folding forward does not. Shoulder width is
the one span that survives both - shoulders keep their width through a slump,
so a change in it is distance and not posture.

*A chin rest is an arm fact, not a head fact.* The head barely moves; a hand
arrives beside it. So it is read as a wrist near the head with the elbow
dropped below the shoulder - the folded arm - and it is checked before the
head-drop rules, because resting your chin lowers the head slightly and would
otherwise read as the beginning of a slump.

*Every threshold is relative to a captured reference.* Absolute pixels would
mean recalibrating for each person, each chair height and each camera distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# MediaPipe pose landmark indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

UPRIGHT = "UPRIGHT"
RECLINE = "RECLINE"
SLUMP = "SLUMP"
CHIN_REST = "CHIN_REST"
ABSENT = "ABSENT"

# All distances below are in shoulder widths, so these are ratios, not pixels.
#
# The two head-drop numbers are far apart on purpose. Dividing by shoulder width
# cancels distance: move the whole person away and numerator and denominator
# shrink together, leaving head_drop where it was. So leaning back does not
# register as "head down" the way it looks like it should - all that survives
# the normalisation is the neck foreshortening as the torso tips, which is a
# third of what folding onto the desk produces. One threshold cannot catch both.
SLUMP_DROP_ON = 0.35    # folding forward brings the nose down to the shoulder line
SLUMP_DROP_OFF = 0.20   # hysteresis
RECLINE_TIP = 0.12      # a tip this slight is all reclining leaves behind
SHRINK_ON = 0.90        # shoulders this much narrower than reference = further off
SHRINK_OFF = 0.95       # hysteresis
CHIN_NEAR = 0.70        # wrist within this of the head counts as beside it
CHIN_RELEASE = 0.90     # and must leave by this much to stop counting
HOLD_SECONDS = 1.5      # a posture must persist this long before it is reported


@dataclass
class Metrics:
    """The scale-free numbers a decision is made from."""
    shoulder_width: float = 0.0
    head_drop: float = 0.0          # (nose_y - shoulder_mid_y) / shoulder_width
    head_drop_delta: float = 0.0    # ... minus the reference value
    scale_ratio: float = 1.0        # shoulder width / reference shoulder width
    wrist_to_head: float = math.inf  # nearest wrist, in shoulder widths
    arm_folded: bool = False


@dataclass
class State:
    label: str = ABSENT
    candidate: str = ABSENT         # what it looks like right now, before the hold
    held_for: float = 0.0
    metrics: Metrics = field(default_factory=Metrics)
    has_reference: bool = False
    note: str = ""


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def measure(pts) -> Metrics | None:
    """Reduce the landmarks to the handful of ratios the rules use.

    Returns None when both shoulders are not visible: shoulder width is the
    ruler everything else is expressed in, and with one shoulder there is no
    midpoint to measure the head against either.
    """
    if L_SHOULDER not in pts or R_SHOULDER not in pts or NOSE not in pts:
        return None

    ls, rs, nose = pts[L_SHOULDER], pts[R_SHOULDER], pts[NOSE]
    width = _dist(ls, rs)
    if width < 1e-6:
        return None

    mid_y = (ls[1] + rs[1]) / 2.0
    m = Metrics(shoulder_width=width, head_drop=(nose[1] - mid_y) / width)

    near = math.inf
    folded = False
    for wrist, elbow, shoulder in ((L_WRIST, L_ELBOW, L_SHOULDER),
                                   (R_WRIST, R_ELBOW, R_SHOULDER)):
        if wrist not in pts:
            continue
        d = _dist(pts[wrist], nose) / width
        if d < near:
            near = d
            # y grows downward, so an elbow below its shoulder is a folded arm -
            # propped on a desk. A hand raised to wave passes near the head too,
            # but with the elbow up.
            folded = elbow in pts and pts[elbow][1] > pts[shoulder][1]
    m.wrist_to_head = near
    m.arm_folded = folded
    return m


class PostureTracker:
    def __init__(self, hold_seconds: float = HOLD_SECONDS):
        self.hold = hold_seconds
        self.ref: Metrics | None = None
        self._label = ABSENT
        self._seen = ABSENT             # candidate the hold timer is running on
        self._since: float | None = None
        self._was_dropped = False
        self._was_chin = False
        self._was_receded = False

    # -- reference ------------------------------------------------------

    def capture_reference(self, pts) -> bool:
        m = measure(pts)
        if m is None:
            return False
        self.ref = m
        self._was_dropped = self._was_chin = self._was_receded = False
        return True

    # -- per frame ------------------------------------------------------

    def update(self, pts, now: float) -> State:
        m = measure(pts) if pts else None
        if m is None:
            self._commit(ABSENT, now)
            return State(self._label, ABSENT, self._held(now), Metrics(),
                         self.ref is not None, "no shoulders in frame")

        if self.ref is None:
            self._label = UPRIGHT
            return State(UPRIGHT, UPRIGHT, 0.0, m, False, "no reference captured")

        m.head_drop_delta = m.head_drop - self.ref.head_drop
        m.scale_ratio = m.shoulder_width / self.ref.shoulder_width

        candidate, note = self._classify(m)

        self._commit(candidate, now)
        return State(self._label, candidate, self._held(now), m, True, note)

    # -- rules ----------------------------------------------------------

    def _classify(self, m: Metrics):
        # Hysteresis on both switches. A wrist hovering at the threshold, or a
        # head bobbing across it, would otherwise alternate labels every frame
        # and no hold timer would ever fill.
        near = CHIN_RELEASE if self._was_chin else CHIN_NEAR
        self._was_chin = m.wrist_to_head < near and m.arm_folded

        drop_gate = SLUMP_DROP_OFF if self._was_dropped else SLUMP_DROP_ON
        self._was_dropped = m.head_drop_delta > drop_gate

        shrink_gate = SHRINK_OFF if self._was_receded else SHRINK_ON
        self._was_receded = m.scale_ratio < shrink_gate

        if self._was_chin and not self._was_dropped:
            return CHIN_REST, "wrist beside the head, elbow dropped"

        # Reclining is read from the shoulders receding, not from the head
        # falling - see the thresholds above for why the fall is small. The tip
        # has to be there too: shoulders that narrow with the neck unchanged is
        # a chair pushed back, which is not a posture.
        if self._was_receded and m.head_drop_delta > RECLINE_TIP:
            return RECLINE, (f"shoulders {1 - m.scale_ratio:.0%} narrower, "
                             f"neck foreshortened")

        if self._was_dropped:
            return SLUMP, "head down to the shoulder line, shoulders where they were"

        return UPRIGHT, ""

    # -- hold timer -----------------------------------------------------

    def _commit(self, candidate: str, now: float):
        """Only report a posture once it has stayed put.

        At the bottom of a nod the geometry is a slump; the two differ in time,
        not in shape. Requiring the candidate to persist is what separates them,
        and it keeps a single bad frame from flipping the label.
        """
        if candidate != self._seen:
            self._seen = candidate
            self._since = now
        if self._since is not None and now - self._since >= self.hold:
            self._label = candidate

    def _held(self, now: float) -> float:
        return 0.0 if self._since is None else max(0.0, now - self._since)
