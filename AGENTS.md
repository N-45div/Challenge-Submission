# Agent Notes

This Crossing Challenge submission was developed with Codex as an AI coding
assistant.

Relevant workflow:

- Inspected the challenge README, schema, grader, starter baseline, and tests.
- Ran the starter baseline and local grader to establish a dev reference score.
- Replaced constant-velocity-only trajectory prediction with learned residual
  XGBoost regressors over a constant-velocity prior.
- Kept intent as a compact XGBoost classifier because it calibrated better on
  dev log-loss than the larger trajectory feature set.
- Verified the final package with `python grade.py`, `python -m pytest tests/`,
  `docker build`, and Docker grader-mode prediction.

No external APIs, external datasets, pretrained checkpoints, or eval-set data
are used by the final inference path.
