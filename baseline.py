"""Train the Crossing Challenge submission model.

This replaces the starter's intent-only baseline with:
  - calibrated XGBoost intent classification
  - XGBoost trajectory residual regression over constant velocity

Run:
    python baseline.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier, XGBRegressor

from grade import ADE_FLOOR, BCE_FLOOR, HORIZONS, score
from predict import HORIZON_KEYS, _constant_velocity_centers, _engineered_features, _intent_features

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
INTENT_LOGIT_BIAS = -0.05
TRAJECTORY_ALPHA = {
    "bbox_500ms": 1.0,
    "bbox_1000ms": 1.0,
    "bbox_1500ms": 0.95,
    "bbox_2000ms": 0.925,
}

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row: pd.Series) -> dict:
    return {k: row[k] for k in REQUEST_FIELDS}


def featurize(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
    return X


def featurize_intent(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _intent_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _intent_features(row_to_request(df.iloc[i]))
    return X


def centers_from_bbox_col(df: pd.DataFrame, col: str) -> np.ndarray:
    boxes = np.stack([np.asarray(x, dtype=np.float64) for x in df[col].to_numpy()])
    return (boxes[:, [0, 1]] + boxes[:, [2, 3]]) * 0.5


def cv_centers(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {k: np.empty((len(df), 2), dtype=np.float64) for k in HORIZON_KEYS}
    for i, (_, row) in enumerate(df.iterrows()):
        req = row_to_request(row)
        _, _, _, _, centers = _constant_velocity_centers(req)
        for key, center in zip(HORIZON_KEYS, centers):
            out[key][i] = center
    return out


def make_pred_frame(
    df: pd.DataFrame,
    intent: np.ndarray,
    cv: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray] | None = None,
    trajectory_alpha: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        hist = np.stack([np.asarray(x, dtype=np.float64) for x in row["bbox_history"]])
        w = float(max(hist[-1, 2] - hist[-1, 0], 1.0))
        h = float(max(hist[-1, 3] - hist[-1, 1], 1.0))
        flat: list[float | str] = [row["ped_id"], float(intent[i])]
        for key in HORIZON_KEYS:
            cx, cy = cv[key][i]
            if residuals is not None:
                alpha = 1.0 if trajectory_alpha is None else trajectory_alpha.get(key, 1.0)
                cx += residuals[f"{key}_dx"][i] * alpha
                cy += residuals[f"{key}_dy"][i] * alpha
            flat.extend([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])
        rows.append(flat)
    return pd.DataFrame(
        rows,
        columns=["ped_id", "intent"] + [f"{h}_{c}" for h in HORIZONS for c in ("x1", "y1", "x2", "y2")],
    )


def train_intent(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray):
    clf = XGBClassifier(
        n_estimators=700,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2.0,
        reg_lambda=4.0,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
    return clf


def train_traj_models(
    X_train: np.ndarray,
    train: pd.DataFrame,
    X_dev: np.ndarray,
    dev: pd.DataFrame,
) -> tuple[dict[str, XGBRegressor], dict[str, np.ndarray]]:
    train_cv = cv_centers(train)
    dev_cv = cv_centers(dev)
    models: dict[str, XGBRegressor] = {}
    dev_residuals: dict[str, np.ndarray] = {}

    for key in HORIZON_KEYS:
        truth_train = centers_from_bbox_col(train, key)
        truth_dev = centers_from_bbox_col(dev, key)
        for axis, idx in (("dx", 0), ("dy", 1)):
            target = truth_train[:, idx] - train_cv[key][:, idx]
            reg = XGBRegressor(
                n_estimators=650,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2.0,
                reg_lambda=6.0,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=100 + len(models),
            )
            reg.fit(X_train, target, eval_set=[(X_dev, truth_dev[:, idx] - dev_cv[key][:, idx])], verbose=False)
            name = f"{key}_{axis}"
            models[name] = reg
            dev_residuals[name] = reg.predict(X_dev)

    return models, dev_residuals


def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    print(f"  positive rates: train {train.will_cross_2s.mean():.3f}, dev {dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    X_train_intent = featurize_intent(train)
    X_dev_intent = featurize_intent(dev)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train.shape}")

    print("\nTraining intent classifier...")
    t0 = time.time()
    intent = train_intent(X_train_intent, y_train, X_dev_intent, y_dev)
    dev_probs = intent.predict_proba(X_dev_intent)[:, 1]
    dev_probs_biased = np.clip(dev_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(dev_probs_biased / (1.0 - dev_probs_biased)) + INTENT_LOGIT_BIAS
    dev_probs_biased = 1.0 / (1.0 + np.exp(-logits))
    ll = log_loss(y_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6))
    ll_biased = log_loss(y_dev, np.clip(dev_probs_biased, 1e-6, 1 - 1e-6))
    prior_ll = log_loss(y_dev, np.full_like(dev_probs, y_train.mean(), dtype=np.float64))
    print(f"  {time.time() - t0:.1f}s")
    print(
        f"  Dev log-loss: {ll:.4f}; biased {ll_biased:.4f}  "
        f"(class-prior {prior_ll:.4f}, term {ll_biased / BCE_FLOOR:.3f})"
    )

    print("\nTraining trajectory residual regressors...")
    t0 = time.time()
    traj, dev_residuals = train_traj_models(X_train, train, X_dev, dev)
    dev_cv = cv_centers(dev)
    print(f"  {time.time() - t0:.1f}s")

    preds = make_pred_frame(dev, dev_probs_biased, dev_cv, dev_residuals, TRAJECTORY_ALPHA)
    s = score(preds, dev)
    print(
        "\nFull Dev score: "
        f"{s['score']:.4f}   "
        f"(intent_term {s['intent_term']:.3f}, traj_term {s['traj_term']:.3f}; "
        f"BCE {s['intent_bce']:.4f}, ADE {s['mean_ade_px']:.1f} px, ADE floor {ADE_FLOOR:.1f})"
    )

    for key in HORIZON_KEYS:
        truth = centers_from_bbox_col(dev, key)
        alpha = TRAJECTORY_ALPHA[key]
        pred = dev_cv[key] + alpha * np.column_stack([dev_residuals[f"{key}_dx"], dev_residuals[f"{key}_dy"]])
        ade = float(np.hypot(pred[:, 0] - truth[:, 0], pred[:, 1] - truth[:, 1]).mean())
        cv_ade = float(np.hypot(dev_cv[key][:, 0] - truth[:, 0], dev_cv[key][:, 1] - truth[:, 1]).mean())
        print(f"  {key}: learned ADE {ade:.1f} px  vs CV {cv_ade:.1f} px")

    print(f"\nSaving model -> {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "intent": intent,
                "traj": traj,
                "intent_logit_bias": INTENT_LOGIT_BIAS,
                "trajectory_alpha": TRAJECTORY_ALPHA,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
