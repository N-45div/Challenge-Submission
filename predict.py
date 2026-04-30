"""Submission entry point for the Crossing Challenge.

The model artifact produced by baseline.py contains:
  - one calibrated intent classifier
  - eight trajectory residual regressors: dx/dy for each future horizon

Trajectory prediction starts from a constant-velocity center extrapolation and
learns a residual correction. The scorer only uses bbox centers for ADE, so the
current bbox width/height is carried forward for stable, cheap inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

TIME_CATS = ("daytime", "nighttime", "n/a", "")
WEATHER_CATS = ("clear", "cloudy", "cloud", "rain", "snow", "n/a", "")
LOCATION_CATS = ("street", "plaza", "indoor", "")

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
    return _cached_model


def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _safe_hist(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])
    if hist.shape != (16, 4):
        hist = np.resize(hist, (16, 4)).astype(np.float64)
    return np.nan_to_num(hist, nan=0.0, posinf=4000.0, neginf=-2000.0)


def _center_motion(req: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    hist = _safe_hist(req)
    fw = max(float(req.get("frame_w", 1920) or 1920), 1.0)
    fh = max(float(req.get("frame_h", 1080) or 1080), 1.0)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = np.maximum(hist[:, 2] - hist[:, 0], 1.0)
    h = np.maximum(hist[:, 3] - hist[:, 1], 1.0)
    return cx, cy, w, h, fw, fh


def _one_hot(value: object, cats: tuple[str, ...]) -> list[float]:
    s = "" if value is None else str(value)
    return [1.0 if s == c else 0.0 for c in cats]


def _engineered_features(req: dict) -> np.ndarray:
    """Feature layout shared by training and inference."""
    cx, cy, w, h, fw, fh = _center_motion(req)
    nx = cx / fw
    ny = cy / fh
    nw = w / fw
    nh = h / fh

    vx = np.diff(cx) / fw
    vy = np.diff(cy) / fh
    ax = np.diff(vx)
    ay = np.diff(vy)
    scale_v = np.diff(np.log(np.maximum(h, 1.0)))

    ego_s = np.asarray(req.get("ego_speed_history", [0.0] * 16), dtype=np.float64)
    ego_y = np.asarray(req.get("ego_yaw_history", [0.0] * 16), dtype=np.float64)
    if ego_s.size != 16:
        ego_s = np.resize(ego_s, 16)
    if ego_y.size != 16:
        ego_y = np.resize(ego_y, 16)
    ego_s = np.nan_to_num(ego_s, nan=0.0, posinf=0.0, neginf=0.0)
    ego_y = np.nan_to_num(ego_y, nan=0.0, posinf=0.0, neginf=0.0)

    def stats(a: np.ndarray) -> list[float]:
        return [
            float(a[-1]),
            float(a.mean()),
            float(a.std()),
            float(a.min()),
            float(a.max()),
        ]

    feats: list[float] = []
    feats.extend(nx.tolist())
    feats.extend(ny.tolist())
    feats.extend(nw.tolist())
    feats.extend(nh.tolist())
    feats.extend(vx.tolist())
    feats.extend(vy.tolist())
    feats.extend(ax.tolist())
    feats.extend(ay.tolist())
    feats.extend(scale_v.tolist())
    feats.extend(stats(vx))
    feats.extend(stats(vy))
    feats.extend(stats(ax) if ax.size else [0.0] * 5)
    feats.extend(stats(ay) if ay.size else [0.0] * 5)
    feats.extend(stats(nw))
    feats.extend(stats(nh))
    feats.extend([
        float((h[-1] / (w[-1] + 1e-6))),
        float((h.mean() / (w.mean() + 1e-6))),
        float(np.hypot(vx[-4:].mean(), vy[-4:].mean())),
        float(np.hypot(vx[-8:].mean(), vy[-8:].mean())),
        float(np.hypot(vx, vy).std()),
        float(req.get("ego_available", False)),
        float(req.get("requested_at_frame", 0) or 0) / 30000.0,
        float((int(req.get("requested_at_frame", 0) or 0) % 30) / 30.0),
    ])
    feats.extend((ego_s / 30.0).tolist())
    feats.extend(ego_y.tolist())
    feats.extend(stats(ego_s / 30.0))
    feats.extend(stats(ego_y))
    feats.extend(_one_hot(req.get("time_of_day"), TIME_CATS))
    feats.extend(_one_hot(req.get("weather"), WEATHER_CATS))
    feats.extend(_one_hot(req.get("location"), LOCATION_CATS))

    arr = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)


def _intent_features(req: dict) -> np.ndarray:
    """Compact starter-style intent features; these calibrated best on dev."""
    cx, cy, w, h, fw, fh = _center_motion(req)
    vx = np.diff(cx)
    vy = np.diff(cy)

    ego_s = np.asarray(req.get("ego_speed_history", [0.0] * 16), dtype=np.float64)
    ego_y = np.asarray(req.get("ego_yaw_history", [0.0] * 16), dtype=np.float64)
    if ego_s.size != 16:
        ego_s = np.resize(ego_s, 16)
    if ego_y.size != 16:
        ego_y = np.resize(ego_y, 16)
    ego_s = np.nan_to_num(ego_s, nan=0.0, posinf=0.0, neginf=0.0)
    ego_y = np.nan_to_num(ego_y, nan=0.0, posinf=0.0, neginf=0.0)

    feats = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx.std() / fw,
        vy.std() / fh,
        (h / (w + 1e-6)).mean(),
        float(req.get("ego_available", False)),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]
    arr = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)


def _constant_velocity_centers(req: dict) -> tuple[float, float, float, float, list[tuple[float, float]]]:
    cx, cy, w, h, _, _ = _center_motion(req)
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])
    centers = [(cur_cx + vx * t, cur_cy + vy * t) for t in HORIZONS_FRAMES]
    return cur_cx, cur_cy, float(w[-1]), float(h[-1]), centers


def predict(request: dict) -> dict:
    model = _load_model()
    feats = _engineered_features(request).reshape(1, -1)
    intent_feats = _intent_features(request).reshape(1, -1)

    intent_prob = float(model["intent"].predict_proba(intent_feats)[0, 1])
    intent_prob = float(np.clip(np.nan_to_num(intent_prob, nan=0.5), 1e-6, 1.0 - 1e-6))

    _, _, w_last, h_last, centers = _constant_velocity_centers(request)
    out: dict[str, list[float] | float] = {"intent": intent_prob}
    for i, key in enumerate(HORIZON_KEYS):
        nx, ny = centers[i]
        if "traj" in model:
            dx = float(model["traj"][f"{key}_dx"].predict(feats)[0])
            dy = float(model["traj"][f"{key}_dy"].predict(feats)[0])
            nx += np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            ny += np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = [
            float(nx - w_last * 0.5),
            float(ny - h_last * 0.5),
            float(nx + w_last * 0.5),
            float(ny + h_last * 0.5),
        ]
    return out
