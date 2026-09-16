"""54x42 zone 맵과 자세 판정을 한 화면에 띄운다.

왼쪽: zone 거리 배열을 열화상 팔레트로 그린 것(가까울수록 뜨겁게).
오른쪽: 현재 자세 라벨과 FSM 으로 나갈 phi/delta, 그리고 판정 근거가 된 기하 특징.

진행 순서
  1) 배경 캘리브레이션 — 책상을 비우고 SPACE. 책상·벽의 zone 별 기준 거리를 잡는다.
  2) 자세 baseline    — 바른 자세로 앉아서 SPACE. 이후 판정은 이 기준 대비 상대값이다.
  3) 판정             — UPRIGHT / SLUMP(엎드림) / RECLINE(젖힘) / DROWSY(졸음) / ABSENT

렌더링과 판정은 zone 배열만 본다. 카메라는 tof_stub 이 zone 배열을 만들 때만 쓰이고,
실센서(VL53L9CX)로 갈아끼우면 이 파일은 그대로다.

Keys:
    SPACE  현재 단계 진행       n  배경 다시 캘리브레이션
    b      자세 baseline 다시   c  팔레트 순환      g  zone 격자 on/off
    1/2/3/4  실측 라벨 기록 (upright/slump/recline/drowsy)   0  기록 정지
    p      스냅샷 저장          q / ESC  종료
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from posture import ZONE_COLS, ZONE_ROWS, PostureTracker, PostureVerdict
from palette import PALETTES, sensor_grid

LABEL_COLOR = {
    "UPRIGHT": (110, 220, 110),
    "SLUMP": (70, 70, 245),
    "DROWSY": (200, 120, 255),
    "RECLINE": (60, 210, 245),
    "ABSENT": (150, 150, 150),
    "BASELINE": (240, 200, 90),
    "UNKNOWN": (180, 180, 180),
}
PANEL_W = 340
PANEL_MIN_H = 687    # 패널 내용이 다 들어가는 최소 높이. 더 짧으면 푸터가 겹친다.
TAG_KEYS = {ord("1"): "upright", ord("2"): "slump", ord("3"): "recline",
            ord("4"): "drowsy"}
NEAR_MM, FAR_MM = 450.0, 2600.0

STEP_BACKGROUND, STEP_BASELINE, STEP_LIVE = range(3)
STEP_PROMPT = {
    STEP_BACKGROUND: ("STEP 1  background", "Clear the desk, then press SPACE"),
    STEP_BASELINE: ("STEP 2  posture baseline", "Sit upright, then press SPACE"),
    STEP_LIVE: ("", ""),
}


def render_zones(depth_mm: np.ndarray, palette: int, cell: int, grid: bool) -> np.ndarray:
    """zone 거리 배열 -> 열화상 이미지. 가까울수록 뜨겁다."""
    d = np.where(np.isfinite(depth_mm), depth_mm, FAR_MM)
    heat = 1.0 - np.clip((d - NEAR_MM) / (FAR_MM - NEAR_MM), 0.0, 1.0)
    u8 = (heat * 255.0).astype(np.uint8)

    _, cmap = PALETTES[palette]
    small = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if cmap is None else cv2.applyColorMap(u8, cmap)
    out = cv2.resize(small, (ZONE_COLS * cell, ZONE_ROWS * cell), interpolation=cv2.INTER_NEAREST)
    if grid:
        sensor_grid(out, cell)
    return out


def _text(img, s, org, scale=0.5, color=(235, 235, 235), weight=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _bar(img, org, width, value, color):
    x, y = org
    cv2.rectangle(img, (x, y), (x + width, y + 11), (55, 55, 60), -1)
    filled = int(width * float(np.clip(value, 0.0, 1.0)))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + 11), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + 11), (95, 95, 100), 1)


def render_panel(verdict: PostureVerdict, height: int, *, step: int,
                 bg_ready: bool, base_ready: bool, fps: float) -> np.ndarray:
    p = np.full((height, PANEL_W, 3), 26, np.uint8)
    f = verdict.features
    color = LABEL_COLOR.get(verdict.label, (200, 200, 200))
    bar_w = PANEL_W - 32

    title, hint = STEP_PROMPT[step]
    if title:
        _text(p, title, (16, 26), 0.5, (240, 200, 90))
        _text(p, hint, (16, 46), 0.42, (200, 200, 205))
        y0 = 74
    else:
        y0 = 30

    _text(p, "POSTURE", (16, y0), 0.45, (150, 150, 155))
    _text(p, verdict.label, (16, y0 + 40), 1.1, color, 2)
    if verdict.note:
        _text(p, verdict.note, (16, y0 + 62), 0.42, (165, 165, 170))
    _text(p, f"present  {'yes' if verdict.present else 'no'}", (16, y0 + 84), 0.44,
          (110, 220, 110) if verdict.present else (150, 150, 150))

    y = y0 + 116
    _text(p, "-> FSM posture Signal", (16, y), 0.44, (150, 150, 155))
    for key, val, col in (("phi   (focus)", verdict.phi, (200, 190, 110)),
                          ("delta (fatigue)", verdict.delta, (90, 130, 245))):
        y += 24
        _text(p, key, (16, y), 0.44)
        _text(p, f"{val:.2f}", (PANEL_W - 58, y), 0.44, col)
        _bar(p, (16, y + 6), bar_w, val, col)
        y += 18

    if verdict.parts:
        y += 22
        _text(p, "deviation", (16, y), 0.44, (150, 150, 155))
        for key, col in (("slump", (70, 70, 245)), ("recline", (60, 210, 245)),
                         ("drowsy", (200, 120, 255))):
            val = verdict.parts.get(key, 0.0)
            y += 24
            _text(p, key, (16, y), 0.44)
            _text(p, f"{val:.2f}", (PANEL_W - 58, y), 0.44, col)
            _bar(p, (16, y + 6), bar_w, val, col)
            y += 18

    y += 22
    _text(p, "zone geometry", (16, y), 0.44, (150, 150, 155))
    head = "--" if not np.isfinite(f.head_mm) else f"{f.head_mm:.0f} mm"
    # dist vs base 가 엎드림/젖힘을 가르는 유일한 축이라 눈에 띄게 둔다.
    dd = verdict.parts.get("dist_mm")
    dist_txt = "--" if dd is None else f"{dd:+.0f} mm"
    for key, val in (("occupancy", f"{f.occupancy * 100:.1f} %"),
                     ("head row", f"{f.top_row:.3f}"),
                     ("spread", f"{f.spread:.3f}"),
                     ("head dist", head),
                     ("dist vs base", dist_txt),
                     ("motion", f"{f.motion * 100:.1f} %"),
                     ("nod / min", f"{verdict.nod_rate:.1f}")):
        y += 21
        _text(p, key, (16, y), 0.43, (175, 175, 180))
        _text(p, val, (PANEL_W - 122, y), 0.43)

    _text(p, f"background {'captured' if bg_ready else 'auto-learning'}", (16, height - 54), 0.43,
          (110, 220, 110) if bg_ready else (60, 210, 245))
    _text(p, f"baseline   {'ok' if base_ready else 'not set'}", (16, height - 36), 0.43,
          (110, 220, 110) if base_ready else (60, 210, 245))
    _text(p, f"{fps:4.1f} fps   SPACE next  n bg  b base  q quit", (16, height - 14), 0.4,
          (140, 140, 145))
    return p


def compose(depth_mm, verdict, *, step=STEP_LIVE, palette=0, cell=14, grid=False,
            bg_ready=False, base_ready=False, fps=0.0) -> np.ndarray:
    zones = render_zones(depth_mm, palette, cell, grid)
    height = max(zones.shape[0], PANEL_MIN_H)
    if zones.shape[0] < height:            # zone 맵이 짧으면 위아래로 여백을 준다
        pad = height - zones.shape[0]
        top = pad // 2
        zones = cv2.copyMakeBorder(zones, top, pad - top, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(18, 18, 20))
    panel = render_panel(verdict, height, step=step, bg_ready=bg_ready,
                         base_ready=base_ready, fps=fps)
    return np.hstack([zones, panel])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("camera", "esp"), default="camera",
                    help="zone 배열을 어디서 받을지. 판정·화면은 어느 쪽이든 같다")
    ap.add_argument("--camera", type=int, default=0, help="--source camera")
    ap.add_argument("--port", help="--source esp: 시리얼 포트 (예: COM8)")
    ap.add_argument("--baud", type=int, default=921600, help="--source esp: LINK_BAUD 와 맞출 것")
    ap.add_argument("--cell", type=int, default=14, help="zone 하나를 화면 몇 픽셀로 그릴지")
    ap.add_argument("--bg-frames", type=int, default=30)
    ap.add_argument("--baseline-frames", type=int, default=60)
    ap.add_argument("--outdir", default="snapshots")
    ap.add_argument("--log", default="posture_log.csv",
                    help="1/2/3 으로 자세를 표시하는 동안 특징을 여기 기록한다")
    args = ap.parse_args()

    # 센서 의존은 여기서만 들어온다. 둘 다 read() -> ZoneFrame 이라 아래 루프는
    # 어느 쪽인지 알 필요가 없다.
    if args.source == "esp":
        if not args.port:
            ap.error("--source esp 는 --port 가 필요하다 (예: --port COM8)")
        from esp_source import EspZoneSource
        stub = EspZoneSource(args.port, args.baud)
    else:
        from camera_source import CameraZoneSource
        stub = CameraZoneSource(camera=args.camera)
    tracker = PostureTracker()
    step = STEP_BACKGROUND
    print("STEP 1 — 책상을 비우고 창에서 SPACE 를 누르세요.")

    outdir = Path(args.outdir)
    palette, grid = 0, False
    fps, last = 0.0, time.perf_counter()
    tag, log_rows = None, []
    try:
        while True:
            zone = stub.read()
            if zone is None:
                print("프레임 취득 실패, 종료합니다.")
                break
            now = time.perf_counter()
            verdict = tracker.update(zone, now)

            # 수집이 끝나면 다음 단계로 넘어간다.
            if step == STEP_BACKGROUND and tracker.background.captured:
                step = STEP_BASELINE
                print("STEP 2 — 바른 자세로 앉아서 SPACE 를 누르세요.")
            elif step == STEP_BASELINE and tracker.baseline is not None:
                step = STEP_LIVE
                print("판정 시작. 엎드리거나 뒤로 젖혀 보세요.")

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            # 판정 구간은 무조건 기록한다. 키를 눌러야만 남기게 했더니 실제 실행에서
            # 매번 빠졌다. 1~4 로 붙이는 정답 라벨(tag)은 선택 사항으로 둔다.
            if step == STEP_LIVE:
                fe = verdict.features
                log_rows.append({
                    "tag": tag or "", "t": f"{now:.3f}", "label": verdict.label,
                    "occupancy": f"{fe.occupancy:.4f}", "top_row": f"{fe.top_row:.4f}",
                    "spread": f"{fe.spread:.4f}", "head_mm": f"{fe.head_mm:.1f}",
                    "dist_mm": f"{verdict.parts.get('dist_mm', 0.0):.1f}",
                    "slump": f"{verdict.parts.get('slump', 0.0):.3f}",
                    "recline": f"{verdict.parts.get('recline', 0.0):.3f}",
                    "drowsy": f"{verdict.parts.get('drowsy', 0.0):.3f}",
                    "nod_rate": f"{verdict.nod_rate:.2f}",
                    "phi": f"{verdict.phi:.3f}", "delta": f"{verdict.delta:.3f}",
                })

            canvas = compose(zone.depth_mm, verdict, step=step, palette=palette,
                             cell=args.cell, grid=grid,
                             bg_ready=tracker.background.captured,
                             base_ready=tracker.baseline is not None, fps=fps)
            if tag:
                _text(canvas, f"REC {tag}  ({len(log_rows)})", (16, 28), 0.6, (70, 70, 245), 2)
            cv2.imshow("posture from zone map", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if step == STEP_BACKGROUND:
                    tracker.start_background(args.bg_frames)
                    print("배경 수집 중... 책상을 비워 두세요.")
                elif step == STEP_BASELINE:
                    tracker.start_baseline(args.baseline_frames)
                    print("자세 baseline 수집 중... 바른 자세를 유지하세요.")
            elif key == ord("n"):
                step = STEP_BACKGROUND
                tracker.background.captured = False
                print("STEP 1 다시 — 책상을 비우고 SPACE.")
            elif key == ord("b"):
                step = STEP_BASELINE
                tracker.baseline = None
                print("STEP 2 다시 — 바른 자세로 앉아서 SPACE.")
            elif key in TAG_KEYS:
                tag = TAG_KEYS[key]
                print(f"기록 시작: {tag} (0 누르면 정지)")
            elif key == ord("0"):
                tag = None
                print("기록 정지")
            elif key == ord("c"):
                palette = (palette + 1) % len(PALETTES)
            elif key == ord("g"):
                grid = not grid
            elif key == ord("p"):
                outdir.mkdir(parents=True, exist_ok=True)
                path = outdir / f"posture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), canvas)
                print(f"saved {path}")
    finally:
        stub.release()
        cv2.destroyAllWindows()
        if log_rows:
            import csv
            with open(args.log, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
                w.writeheader()
                w.writerows(log_rows)
            print(f"{len(log_rows)} 행 기록: {args.log}")


if __name__ == "__main__":
    main()
