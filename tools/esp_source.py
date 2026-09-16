"""ESP32-CAM(+RP2040 브리지)을 zone 소스로 쓰는 어댑터.

`tof_stub.py` 와 같은 자리다 — `read()` 가 `ZoneFrame` 을 내놓기만 하면
`posture.py` 와 `posture_viewer.py` 는 손대지 않는다. 센서만 갈아끼우는 구조라
웹캠 스텁 / ESP32-CAM / 실 ToF 가 전부 같은 경계를 쓴다.

    tof_stub.CameraToFStub  웹캠 + MediaPipe -> zone 거리
    esp_source.EspZoneSource  ESP32-CAM 차분 필드 -> zone 거리   <- 이 파일
    (나중) VL53L9CX 드라이버  실제 거리

**ESP 는 거리를 안 준다.** 밝기 차분을 54x42 로 줄인 coverage 와, 그걸 이진화·
정제한 mask 만 온다. 그래서 거리는 **겉보기 크기**로 추정한다 — 멀어지면 작게
찍히는 원근을 쓰는 것이고, 웹캠 스텁이 어깨 너비로 하던 것과 같은 원리다.
몸통 폭을 쓰는 이유도 같다: 엎드려도 어깨 너비는 거의 안 변하고 거리만 움직인다.

주의: 이렇게 만든 거리는 기하만 맞고 절대값은 근사다. 엎드림/젖힘을 가르는 데는
충분하지만(부호가 갈리므로), mm 단위 자체에 의미를 두면 안 된다.
"""
from __future__ import annotations

import binascii
import struct
import sys
import threading
import time

import numpy as np

from posture import MAX_RANGE_MM, ZONE_COLS, ZONE_ROWS, ZoneFrame

MAGIC = b"\xA5\x5A"
HDR_LEN = 12
MAX_PAYLOAD = 64 * 1024
TYPE_COVERAGE, TYPE_MASK, TYPE_SKELETON, TYPE_PREVIEW = 1, 2, 3, 4
TYPE_RAW, TYPE_GRAPH = 6, 7
IMAGE_TYPES = {TYPE_COVERAGE, TYPE_MASK, TYPE_SKELETON, TYPE_PREVIEW, TYPE_RAW}

# 앉은 사람의 몸통 폭이 화면 폭의 이 비율일 때 이 거리라고 본다.
REF_SCALE = 0.24
REF_MM = 700.0
BACKGROUND_MM = 2400.0    # 사람이 아닌 zone 에 넣을 거리(벽)
MIN_SCALE = 0.02


def _crc16(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)


class _Link(threading.Thread):
    """시리얼에서 프레임을 뽑아 타입별 최신본만 들고 있는다."""

    daemon = True

    def __init__(self, port: str, baud: int):
        super().__init__()
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.frames: dict[int, np.ndarray] = {}
        self.stamps: dict[int, float] = {}
        self.running = True

    def run(self) -> None:
        import serial
        while self.running:
            try:
                chunk = self.ser.read(8192)
            except serial.SerialException as exc:
                print(f"[serial] {exc}", file=sys.stderr)
                self.running = False
                return
            if chunk:
                self.buf += chunk
                self._parse()

    def send(self, line: str) -> None:
        try:
            self.ser.write((line + "\n").encode())
        except Exception:
            pass

    def stop(self) -> None:
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass

    def _parse(self) -> None:
        buf = self.buf
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                # 프레임이 아닌 건 전부 로그 텍스트다. 헤더 앞부분일 수 있는
                # 마지막 한 바이트만 남긴다.
                keep = 1 if buf[-1:] == b"\xA5" else 0
                if len(buf) > keep:
                    self._text(bytes(buf[: len(buf) - keep]))
                    del buf[: len(buf) - keep]
                return
            if i:
                self._text(bytes(buf[:i]))
                del buf[:i]
            if len(buf) < HDR_LEN:
                return

            typ, _seq, w, h, ln, crc = struct.unpack_from("<BBHHHH", buf, 2)
            known = typ in IMAGE_TYPES or typ == TYPE_GRAPH
            if (not known or ln == 0 or ln > MAX_PAYLOAD
                    or (typ in IMAGE_TYPES and w * h != ln)):
                del buf[:2]
                continue
            if len(buf) < HDR_LEN + ln:
                return

            payload = bytes(buf[HDR_LEN: HDR_LEN + ln])
            if _crc16(payload) != crc:
                del buf[:2]
                continue
            del buf[: HDR_LEN + ln]
            if typ in IMAGE_TYPES:
                with self.lock:
                    self.frames[typ] = np.frombuffer(payload, np.uint8).reshape(h, w)
                    self.stamps[typ] = time.monotonic()

    @staticmethod
    def _text(raw: bytes) -> None:
        sys.stdout.write(raw.decode("utf-8", "replace"))
        sys.stdout.flush()

    def snapshot(self) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        with self.lock:
            return dict(self.frames), dict(self.stamps)


def scale_of(mask: np.ndarray) -> float:
    """몸통 폭 / 화면 폭. 없으면 0."""
    widths = mask.sum(axis=1)
    if not widths.any():
        return 0.0
    wide = widths[widths >= max(widths.max() * 0.5, 1)]
    return float(np.median(wide)) / mask.shape[1]


def mask_to_zone(mask: np.ndarray) -> ZoneFrame:
    """이진 마스크 -> zone 거리 배열. 거리는 겉보기 크기에서 추정한다."""
    person = mask.astype(bool)
    depth = np.full(person.shape, BACKGROUND_MM, np.float64)
    scale = scale_of(person)
    if person.any() and scale >= MIN_SCALE:
        depth[person] = float(np.clip(REF_MM * REF_SCALE / scale, 200.0, MAX_RANGE_MM))
    return ZoneFrame(depth_mm=depth)


class EspZoneSource:
    """tof_stub.CameraToFStub 과 같은 인터페이스."""

    def __init__(self, port: str, baud: int = 921600, stale_s: float = 1.0):
        self._link = _Link(port, baud)
        self._link.start()
        self._stale_s = stale_s
        self._blank = np.zeros((ZONE_ROWS, ZONE_COLS), np.uint8)
        self.coverage: np.ndarray | None = None   # 화면 표시용, 판정에는 안 쓴다
        self.stale = True

    def read(self) -> ZoneFrame | None:
        if not self._link.running:
            return None
        frames, stamps = self._link.snapshot()
        mask = frames.get(TYPE_MASK)
        self.coverage = frames.get(TYPE_COVERAGE)
        self.stale = mask is None or time.monotonic() - stamps.get(TYPE_MASK, 0.0) > self._stale_s
        return mask_to_zone(self._blank if mask is None else mask)

    def send(self, line: str) -> None:
        """RP2040/ESP 에 명령을 그대로 보낸다 (t<n> 임계, b 배경 재학습 등)."""
        self._link.send(line)

    def release(self) -> None:
        self._link.stop()
