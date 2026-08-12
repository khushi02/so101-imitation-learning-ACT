# Results

- **`success_rates.csv`** — one row per evaluation trial: `episode_index` cross-references the recorded
  `khushiiw/so101-clip-bowl-eval` rollout dataset; score `reached` / `grasped` / `placed` as 1/0 during each
  reset window. The aggregate goes into the top-level README's results table.
- **`training_curve.png` / `training_curve_logy.png`** — rendered by `scripts/plot_metrics.py` from
  `training_metrics.csv` (run `python scripts/plot_metrics.py` to rebuild them).
- **`training_metrics.csv`** — the scraped loss points; (re)built from a raw `lerobot-train` log by
  `scripts/plot_training.py`.
- **`media/`** — GIFs and images embedded in the README (hero rollout, a failure case, rig photo, camera views,
  dataset sample).

Raw datasets and trained policies live on the Hugging Face Hub, not in git:
- Dataset: [`khushiiw/so101-clip-bowl`](https://huggingface.co/datasets/khushiiw/so101-clip-bowl)
- Policy: [`khushiiw/act-clip-bowl`](https://huggingface.co/khushiiw/act-clip-bowl)
