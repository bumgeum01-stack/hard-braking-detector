"""모델 로딩과 추론 로직. 웹 프레임워크와 무관하게 동작하도록 분리했다."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import extract_features, preprocess_frames

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hard_braking_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
SAMPLE_DIR = ROOT / "samples"

MAX_UPLOAD_BYTES = 60 * 1024 * 1024   # 60MB (AI Hub CAN 파일은 보통 5~40MB)
MIN_SECONDS = 10                      # 이보다 짧으면 판단할 근거가 부족하다
REQUIRED_MICOM_KEYS = ["gps_vss", "rpm_tick", "ad_brake", "brake_pressure"]


class InputError(ValueError):
    """사용자가 올린 파일에 문제가 있을 때. 메시지를 그대로 화면에 보여준다."""


class Predictor:
    """학습한 모델 파일을 한 번만 읽어 두고 요청마다 재사용한다."""

    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.model = joblib.load(model_path)
        self.features = self.meta["features"]
        self.threshold = float(self.meta.get("threshold", 0.5))

    # ---------- 입력 검증 ----------
    @staticmethod
    def parse_frames(raw_bytes):
        if not raw_bytes:
            raise InputError("빈 파일입니다. CAN JSON 파일을 올려주세요.")
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise InputError(f"파일이 너무 큽니다. {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 이하 파일만 분석할 수 있습니다.")
        try:
            data = json.loads(raw_bytes.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise InputError("텍스트로 읽을 수 없는 파일입니다. JSON 파일이 맞는지 확인해 주세요.")
        except json.JSONDecodeError as e:
            raise InputError(f"JSON 형식이 올바르지 않습니다. ({e.lineno}번째 줄 근처)")

        if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
            raise InputError('CAN 파일 형식이 아닙니다. 최상위에 "frames" 목록이 있어야 합니다.')
        frames = data["frames"]
        if len(frames) < 2:
            raise InputError("프레임이 너무 적습니다. 주행 기록이 담긴 파일인지 확인해 주세요.")

        for i, f in enumerate(frames):
            if not isinstance(f, dict) or "frm_time" not in f or not isinstance(f.get("aim_micom"), dict):
                raise InputError(f'{i + 1}번째 프레임에 "frm_time" 또는 "aim_micom" 항목이 없습니다.')
            missing = [k for k in REQUIRED_MICOM_KEYS if k not in f["aim_micom"]]
            if missing:
                raise InputError(f'{i + 1}번째 프레임의 "aim_micom"에 {", ".join(missing)} 값이 없습니다.')
        return frames

    # ---------- 추론 ----------
    def analyze(self, raw_bytes):
        frames = self.parse_frames(raw_bytes)
        try:
            s1, s5, info = preprocess_frames(frames)
        except (TypeError, ValueError):
            raise InputError("시각·속도·RPM·브레이크 값에 숫자가 아닌 항목이 있습니다.")
        if info["n_seconds"] < MIN_SECONDS:
            raise InputError(f"주행 기록이 {info['n_seconds']}초뿐입니다. {MIN_SECONDS}초 이상이어야 분석할 수 있습니다.")
        if not np.isfinite(s1[["speed", "rpm", "brake_pressure"]].to_numpy(dtype=float)).all():
            raise InputError("시각·속도·RPM·브레이크 값에 숫자가 아닌 항목이 있습니다.")

        feats = extract_features(s1, s5)
        x = pd.DataFrame([feats])[self.features]
        prob = float(self.model.predict_proba(x)[0, 1])

        speed = s1["speed"].to_numpy(dtype=float)
        acc = np.diff(speed) / 3.6
        peak_idx = int(np.argmin(acc)) + 1 if len(acc) else 0
        return {
            "probability": round(prob, 4),
            "hard_braking": bool(prob >= self.threshold),
            "threshold": self.threshold,
            "summary": {
                "duration_sec": int(info["n_seconds"]),
                "speed_max_kmh": round(float(speed.max()), 1),
                "max_decel_ms2": round(float(-feats["dec_1s_min"]), 2),
                "max_decel_g": round(float(-feats["dec_1s_min"]) / 9.81, 2),
                "max_brake_pressure": round(float(feats["bp_max"]), 0),
                "strongest_brake_at_sec": peak_idx,
            },
            "series": {
                "t": list(range(len(s1))),
                "speed": [round(float(v), 1) for v in speed],
                "brake_pressure": [round(float(v), 0) for v in s1["brake_pressure"]],
            },
            "quality": {
                "n_frames": int(info["n_frames"]),
                "interpolated_sec": int(info["n_missing_sec"]),
                "corrected_sec": int(info["n_outlier_sec"]),
            },
            "features": {k: round(float(feats[k]), 3) for k in self.features},
        }

    # ---------- 부가 정보 ----------
    def model_info(self):
        m = self.meta
        return {
            "model_name": m["model_name"],
            "target": m["target"],
            "n_features": len(m["features"]),
            "n_train": m.get("n_train"), "n_val": m.get("n_val"), "n_test": m.get("n_test"),
            "test_metrics": m.get("test_metrics"),
            "dev_cv_metrics": m.get("dev_cv_metrics"),
            "sklearn_version": m.get("sklearn_version"),
            "trained_at": m.get("trained_at"),
        }

    @staticmethod
    def list_samples():
        index_path = SAMPLE_DIR / "index.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text(encoding="utf-8"))

    def analyze_sample(self, sample_id):
        ids = {s["id"] for s in self.list_samples()}
        if sample_id not in ids:
            raise InputError("없는 예시입니다.")
        return self.analyze((SAMPLE_DIR / f"{sample_id}.json").read_bytes())
