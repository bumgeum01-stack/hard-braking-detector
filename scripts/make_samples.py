"""데모용 예시 CAN 파일을 만든다.

AI Hub 원본 데이터는 재배포할 수 없으므로, 같은 파일 형식을 따르는 '합성' 주행 기록을
직접 만들어 저장소에 넣는다. 실제 주행 데이터가 아니며 화면에도 합성 예시라고 표시한다.

실행: python scripts/make_samples.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples"
FPS = 30


def build(speed_1hz, pressure_1hz, seed):
    """1초 단위 속도(km/h)와 브레이크 압력으로 30Hz CAN 프레임 목록을 만든다."""
    rng = np.random.RandomState(seed)
    n = len(speed_1hz)
    t = np.arange(n * FPS) / FPS
    speed = np.interp(t, np.arange(n), speed_1hz)
    pressure = np.interp(t, np.arange(n), pressure_1hz)
    frames = []
    for i in range(len(t)):
        v = max(0.0, speed[i] + rng.normal(0, 0.15))
        p = max(0.0, pressure[i])
        rpm = 750 + v * 22 + rng.normal(0, 15)
        frames.append({
            "frameCnt": f"{i % 1800:04d}",
            "frm_time": str(100000 + int(t[i] * 1000)),
            "aim_micom": {
                "gps_vss": int(round(v)),
                "rpm_tick": int(round(rpm)),
                "ad_brake": int(p > 3),
                "brake_pressure": int(round(p)),
            },
        })
    return {"frames": frames}


def ramp(a, b, sec):
    return list(np.linspace(a, b, sec + 1)[1:])


def calm_cruise():
    v = [40.0] + ramp(40, 60, 20) + [60] * 45 + ramp(60, 45, 15) + [45] * 25 + ramp(45, 55, 14)
    p = [0] * 66 + [18] * 15 + [0] * 39
    return v, p


def hard_braking():
    v = [30.0] + ramp(30, 65, 25) + [65] * 30 + ramp(65, 8, 3) + ramp(8, 0, 2) + [0] * 6 + ramp(0, 50, 22) + [50] * 31
    p = [0] * 56 + [60, 190, 230, 215, 120, 40] + [20] * 5 + [0] * 53
    return v, p


def stop_and_go():
    v = [0.0] * 5 + ramp(0, 45, 14) + [45] * 20 + ramp(45, 0, 12) + [0] * 18 + ramp(0, 50, 15) + [50] * 22 + ramp(50, 20, 13)
    p = [25] * 5 + [0] * 34 + [35] * 12 + [28] * 18 + [0] * 37 + [30] * 14
    return v, p


SAMPLES = [
    ("sample_calm", "정속 주행", "속도를 크게 바꾸지 않고 부드럽게 달리는 주행", calm_cruise, 1),
    ("sample_hard_brake", "급제동 포함", "시속 65km에서 약 3초 만에 거의 멈추는 제동이 한 번 들어 있음", hard_braking, 2),
    ("sample_stop_and_go", "신호 대기 후 출발", "부드럽게 멈췄다가 다시 출발하는 시내 주행", stop_and_go, 3),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    index = []
    for sid, title, desc, fn, seed in SAMPLES:
        v, p = fn()
        n = min(len(v), len(p))
        data = build(v[:n], p[:n], seed)
        (OUT / f"{sid}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        index.append({"id": sid, "title": title, "description": desc, "seconds": n})
        print(sid, n, "sec", len(data["frames"]), "frames")
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
