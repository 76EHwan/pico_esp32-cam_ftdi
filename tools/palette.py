"""zone 배열을 그리는 데 쓰는 팔레트. numpy 와 cv2 말고는 의존이 없다.

`thermal_pose.py` 안에 있던 것을 떼어냈다. 거기엔 MediaPipe import 가 있고 없으면
`sys.exit()` 이라, 팔레트 두 개 때문에 웹캠용 무거운 의존이 딸려 들어왔다. ESP 나
실 ToF 처럼 카메라가 없는 경로에서는 그게 그대로 실행 불가가 된다.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_lut(stops):
    """256칸 BGR 룩업테이블. stops 는 (위치, R, G, B) 를 보간한다."""
    pos = np.array([s[0] for s in stops], np.float32)
    rgb = np.array([s[1] for s in stops], np.float32)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.stack([np.interp(x, pos, rgb[:, ch]) for ch in (2, 1, 0)], axis=1)
    return lut.astype(np.uint8).reshape(256, 1, 3)


# 차가운 쪽이 짙은 파랑이 아니라 검정이다. 비어 있는 zone 이 '차가운 값'이 아니라
# '아무것도 없음'으로 읽히게 하려는 것.
RAINBOW_HC = build_lut([
    (0.00, (0, 0, 0)),
    (0.14, (10, 12, 90)),
    (0.28, (0, 60, 190)),
    (0.40, (0, 130, 150)),
    (0.50, (25, 165, 60)),
    (0.60, (130, 155, 20)),
    (0.70, (205, 45, 25)),
    (0.82, (255, 125, 0)),
    (0.92, (255, 220, 40)),
    (1.00, (255, 255, 205)),
])

PALETTES = [
    ("RAINBOW-HC", RAINBOW_HC),
    ("IRONBOW", cv2.COLORMAP_INFERNO),
    ("RAINBOW", cv2.COLORMAP_JET),
    ("TURBO", cv2.COLORMAP_TURBO),
    ("WHITE-HOT", None),
    ("ARCTIC", cv2.COLORMAP_OCEAN),
]


def sensor_grid(img: np.ndarray, step: int) -> np.ndarray:
    """zone 경계에 어두운 선을 넣어 센서 격자처럼 보이게 한다.

    이음새 행·열만 건드린다. 이미지 전체를 실수로 곱하면 프레임당 수십 ms 가 든다.
    """
    step = max(step, 3)
    img[::step, :] = (img[::step, :] * 0.30).astype(np.uint8)
    img[:, ::step] = (img[:, ::step] * 0.30).astype(np.uint8)
    return img


def zone_grid(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Darken the exact boundaries of a cols x rows grid drawn over `img`.

    sensor_grid takes one integer step, which cannot describe the boundaries
    when the display is not a whole multiple of the cell count: 800 pixels over
    54 cells is 14.81 each, and a step of 14 has drifted three whole cells by
    the right-hand edge - the lines stop matching the blocks and the last column
    falls off the picture.

    cv2.resize with INTER_NEAREST maps destination x to source floor(x*cols/w),
    so cell i begins at ceil(i*w/cols). Computing each boundary that way puts
    every line exactly on a block edge, whatever the two sizes are.
    """
    h, w = img.shape[:2]
    # Integer ceiling division. Computing i*w/cols in floating point and taking
    # ceil puts a boundary one pixel late wherever the quotient is a whole
    # number that lands just above it, which is every time the two sizes share
    # a factor.
    xs = np.unique(((np.arange(cols + 1) * w + cols - 1) // cols).clip(0, w - 1))
    ys = np.unique(((np.arange(rows + 1) * h + rows - 1) // rows).clip(0, h - 1))
    img[ys, :] = (img[ys, :] * 0.30).astype(np.uint8)
    img[:, xs] = (img[:, xs] * 0.30).astype(np.uint8)
    return img
