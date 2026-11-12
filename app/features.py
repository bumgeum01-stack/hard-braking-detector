"""CAN 주행 로그 전처리와 특징 추출.

학습 노트북(notebooks/hard_braking_colab.ipynb)의 전처리·특징 함수와 동일한 코드다.
서비스가 학습 때와 똑같은 방식으로 입력을 가공해야 모델 예측이 의미를 갖는다.
"""
import numpy as np
import pandas as pd

ACC_UP_LIMIT = 25.0     # 1초에 +25km/h (약 0.7g): 승용차가 낼 수 없는 가속 -> GPS 오류로 판단
ACC_DOWN_LIMIT = 40.0   # 1초에 -40km/h (약 1.1g): 타이어 마찰 한계를 넘는 감속 -> 센서 오류로 판단

def frames_to_raw(frames):
    return pd.DataFrame({
        "t_ms": [int(f["frm_time"]) for f in frames],
        "speed": [f["aim_micom"]["gps_vss"] for f in frames],
        "rpm": [f["aim_micom"]["rpm_tick"] for f in frames],
        "brake": [f["aim_micom"]["ad_brake"] for f in frames],
        "brake_pressure": [f["aim_micom"]["brake_pressure"] for f in frames],
    })

def resample_1hz(raw):
    sec = (raw["t_ms"] - raw["t_ms"].iloc[0]) // 1000
    g = raw.groupby(sec)
    s = pd.DataFrame({
        "speed": g["speed"].mean(),
        "rpm": g["rpm"].mean(),
        "brake": g["brake"].max(),
        "brake_pressure": g["brake_pressure"].max(),
    })
    return s.reindex(range(int(s.index.max()) + 1))

def clean_1hz(s):
    s = s.copy()
    n_missing = int(s["speed"].isna().sum())
    s = s.interpolate(limit_direction="both")
    v = s["speed"].to_numpy(dtype=float).copy()
    dv = np.diff(v, prepend=v[0])
    n_outlier = 0
    for i in np.flatnonzero(dv > ACC_UP_LIMIT):
        j = i - 1
        while j >= 0 and v[j] <= 1.0:
            v[j] = np.nan
            n_outlier += 1
            j -= 1
        if j == i - 1:
            v[i] = np.nan
            n_outlier += 1
    for i in np.flatnonzero(dv < -ACC_DOWN_LIMIT):
        v[i] = np.nan
        n_outlier += 1
    if n_outlier:
        s["speed"] = v
        s = s.interpolate(limit_direction="both")
    s["brake"] = s["brake"].round().clip(0, 1)
    return s, {"n_missing_sec": n_missing, "n_outlier_sec": n_outlier}

def resample_5hz(raw):
    """0.2초 단위 시계열. 1초 평균으로는 뭉개지는 순간적인 급제동을 잡기 위해 쓴다."""
    t = raw["t_ms"] - raw["t_ms"].iloc[0]
    g = raw.groupby(t // 200)
    s = pd.DataFrame({"speed": g["speed"].mean(), "brake_pressure": g["brake_pressure"].max()})
    return s.reindex(range(int(s.index.max()) + 1)).interpolate(limit_direction="both")

def preprocess_frames(frames):
    """프레임 목록 -> (1Hz 시계열, 5Hz 시계열, 처리 정보)."""
    raw = frames_to_raw(frames)
    s1, info = clean_1hz(resample_1hz(raw))
    s5 = resample_5hz(raw)
    info["n_frames"] = len(raw)
    info["n_seconds"] = len(s1)
    return s1, s5, info

def _safe(fn, arr, default=0.0):
    return float(fn(arr)) if len(arr) else default

def extract_features(s1, s5):
    """깨끗한 시계열 -> 특징 딕셔너리."""
    v = s1["speed"].to_numpy(dtype=float)              # km/h, 1초 간격
    rpm = s1["rpm"].to_numpy(dtype=float)
    brake = s1["brake"].to_numpy(dtype=float)
    bp1 = s1["brake_pressure"].to_numpy(dtype=float)
    acc = np.diff(v) / 3.6                             # m/s^2
    jerk = np.diff(acc)                                # m/s^3
    acc2 = (v[2:] - v[:-2]) / 3.6 / 2                  # 2초 평균 가속도

    v5 = s5["speed"].to_numpy(dtype=float)             # 0.2초 간격
    bp5 = s5["brake_pressure"].to_numpy(dtype=float)
    dec06 = (v5[3:] - v5[:-3]) / 3.6 / 0.6             # 0.6초 구간 가속도
    dec10 = (v5[5:] - v5[:-5]) / 3.6                   # 1.0초 구간 가속도
    limit = -ACC_DOWN_LIMIT / 3.6
    dec06 = dec06[dec06 >= limit]                      # 물리 한계를 넘는 값은 센서 오류로 보고 제외
    dec10 = dec10[dec10 >= limit]
    bp_roll3 = pd.Series(bp5).rolling(15, min_periods=1).mean().to_numpy()

    # 브레이크를 밟고 있던 구간(에피소드) 중 가장 세게 감속한 구간
    ep = {"ep_speed_drop": 0.0, "ep_mean_decel": 0.0, "ep_pressure_peak": 0.0, "ep_speed_at_onset": 0.0}
    peak, i, n = 0.0, 0, len(v)
    while i < n:
        if brake[i] > 0:
            j = i
            while j + 1 < n and brake[j + 1] > 0:
                j += 1
            start = max(i - 1, 0)
            seg = np.diff(v[start:j + 1])
            pk = -seg.min() / 3.6 if len(seg) else 0.0
            if pk > peak:
                peak = pk
                ep = {"ep_speed_drop": v[start] - v[j], "ep_mean_decel": (v[start] - v[j]) / 3.6 / (j - i + 1),
                      "ep_pressure_peak": bp1[i:j + 1].max(), "ep_speed_at_onset": v[start]}
            i = j + 1
        else:
            i += 1

    launch = (v[3:] - v[:-3])[v[:-3] <= 3]

    return {
        # --- 감속 세기 (1Hz) ---
        "acc_min": _safe(np.min, acc),                       # 1초 최대 감속
        "acc2s_min": _safe(np.min, acc2),                    # 2초 평균 최대 감속
        "acc_p05": _safe(lambda a: np.percentile(a, 5), acc),
        "jerk_min": _safe(np.min, jerk),                     # 감속이 얼마나 갑자기 시작됐나
        "hard_decel_sec": float((acc <= -3.0).sum()),        # -3m/s^2 이하로 감속한 시간(초)
        # --- 감속 세기 (5Hz) ---
        "dec_06s_min": _safe(np.min, dec06),                 # 0.6초 구간 최대 감속
        "dec_1s_min": _safe(np.min, dec10),                  # 1초 구간 최대 감속
        # --- 브레이크 압력 ---
        "bp_max": float(bp5.max()),
        "bp_p95": float(np.percentile(bp5, 95)),
        "bp_rise_02s": _safe(np.max, np.diff(bp5)),          # 0.2초 만에 오른 압력
        "bp_rise_06s": _safe(np.max, bp5[3:] - bp5[:-3]),    # 0.6초 만에 오른 압력
        "bp_sec_over100": float((bp5 >= 100).sum() * 0.2),   # 압력 100 이상 유지 시간(초)
        "bp_sec_over150": float((bp5 >= 150).sum() * 0.2),
        "bp_top3s_mean": float(bp_roll3.max()),              # 가장 세게 밟은 3초의 평균 압력
        # --- 가장 센 제동 구간 ---
        **ep,
        # --- 엔진 ---
        "rpm_drop_max": _safe(lambda a: -np.min(a), np.diff(rpm)),
        # --- 맥락 특징: 모델 입력으로는 쓰지 않고 분석(11단계)에만 쓴다 ---
        "ctx_speed_mean": float(v.mean()),
        "ctx_stop_ratio": float((v <= 1).mean()),
        "ctx_acc_max": _safe(np.max, acc),
        "ctx_launch_dv3_max": max(_safe(np.max, launch), 0.0),
    }

FEATURES = ["acc_min", "acc2s_min", "acc_p05", "jerk_min", "hard_decel_sec", "dec_06s_min", "dec_1s_min",
            "bp_max", "bp_p95", "bp_rise_02s", "bp_rise_06s", "bp_sec_over100", "bp_sec_over150", "bp_top3s_mean",
            "ep_speed_drop", "ep_mean_decel", "ep_pressure_peak", "ep_speed_at_onset", "rpm_drop_max"]
CONTEXT = ["ctx_speed_mean", "ctx_stop_ratio", "ctx_acc_max", "ctx_launch_dv3_max"]
