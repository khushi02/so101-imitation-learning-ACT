#!/usr/bin/env python3
"""Render the training loss curves from the committed metrics CSV.

Usage:
    python scripts/plot_metrics.py

Reads results/training_metrics.csv (step, loss, l1_loss) and writes:
  - results/training_curve.png       (linear y)
  - results/training_curve_logy.png  (log y — shows the late-training detail)

This is the script that produced the charts embedded in the README. It reads the
already-scraped CSV so the figures are reproducible without the multi-GB run log.
To (re)build the CSV from a fresh `lerobot-train` log, use scripts/plot_training.py.

Needs matplotlib: `uv pip install matplotlib`.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "training_metrics.csv"


def load():
    with CSV.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {CSV}")
    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    l1 = [float(r["l1_loss"]) for r in rows]
    return steps, loss, l1


def main():
    steps, loss, l1 = load()

    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib not installed — run `uv pip install matplotlib`.")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Linear
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, loss, label="total loss (L1 + β·KLD)", color="#2563eb", lw=1.8)
    ax.plot(steps, l1, label="L1 action loss", color="#f97316", lw=1.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("ACT training loss — SO-101 clip → bowl (100k steps)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, None)
    fig.tight_layout()
    out = ROOT / "results" / "training_curve.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    # Log y
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2))
    ax2.plot(steps, loss, label="total loss", color="#2563eb", lw=1.8)
    ax2.plot(steps, l1, label="L1 action loss", color="#f97316", lw=1.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("loss (log scale)")
    ax2.set_title("ACT training loss (log y) — SO-101 clip → bowl")
    ax2.legend(frameon=False)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_xlim(0, steps[-1])
    fig2.tight_layout()
    out2 = ROOT / "results" / "training_curve_logy.png"
    fig2.savefig(out2, dpi=150)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
