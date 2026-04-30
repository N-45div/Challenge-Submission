# Crossing Challenge Submission

Pedestrian crossing-intent and 2-second trajectory prediction for the
Gobblecube AI Builder take-home.

## Final Result

| Evaluation | Composite | Intent term | Trajectory term | BCE | ADE |
|---|---:|---:|---:|---:|---:|
| Local 5k Dev sample, `python grade.py` | **0.7192** | 0.843 | 0.596 | 0.2097 | 29.7 px |
| Full Dev | **0.7244** | 0.861 | 0.588 | 0.2141 | 29.3 px |

The starter baseline scored `0.8311` on the same local 5k Dev grader run in
this environment. Lower is better.

## Approach

The starter trajectory path used constant velocity, which was strongest at
short horizons but degraded at +1.5s and +2.0s. I kept constant velocity as a
physics prior and trained residual models to correct it.

The final model has two parts:

- **Intent:** XGBoost classifier on the compact starter-style motion, bbox,
  ego, and scene features. This calibrated better on Dev log-loss than the
  larger trajectory feature set.
- **Trajectory:** eight XGBoost regressors, one `dx` and one `dy` residual for
  each future horizon: +0.5s, +1.0s, +1.5s, and +2.0s. Features include the full
  16-frame bbox history, normalized position, recent velocity, acceleration,
  scale change, ego speed/yaw, frame timing, and scene metadata.

The grader scores bbox center ADE, so inference carries forward the current
bbox width and height and focuses learning capacity on future center position.

## What Improved

Full Dev trajectory ADE by horizon:

| Horizon | Constant velocity | Learned residual |
|---|---:|---:|
| +0.5s | 10.1 px | 7.4 px |
| +1.0s | 23.6 px | 17.1 px |
| +1.5s | 47.3 px | 34.9 px |
| +2.0s | 77.3 px | 57.9 px |

Most of the score gain comes from the long-horizon residual trajectory models.

## Reproduce

```bash
python -m pip install -r requirements.txt
python baseline.py
python grade.py
python -m pytest tests/
docker build -t my-crossing .
docker run --rm -v "$(pwd)/data:/work" my-crossing /work/dev.parquet /work/preds.csv
```

Expected checks:

```text
python grade.py      -> Score: 0.7192
python -m pytest     -> 8 passed
docker image size    -> about 1.31 GB
```

## Repository Contents

- `predict.py`: required row-wise prediction entry point.
- `model.pkl`: trained model artifact used by `predict.py`.
- `Dockerfile`: offline inference container.
- `baseline.py`: training script for the final model.
- `grade.py`: local scoring and grader-mode prediction writer.
- `data/`: provided train/dev parquet files and schema.
- `tests/`: submission contract tests.
- `AGENTS.md`: notes on AI-assisted development workflow.

## External Data

No external datasets, pretrained checkpoints, or external inference APIs are
used. The model is trained only on the provided `data/train.parquet`; Dev is
used only for validation and reporting.

## Next Experiments

Given more time, I would try a small temporal MLP/GRU trained jointly on BCE
and Huber center loss, then calibrate intent probabilities post-hoc. I would
keep the current residual XGBoost model as the reliability baseline because it
is simple, fast, and Docker-safe.
