#!/usr/bin/env python3
"""Parse a LeRobot training log into a loss curve.

Usage:
    python scripts/plot_training.py <run_log.txt> [out.png]

Give it the text you captured from `lerobot-train` (or `hf jobs logs <id> > run_log.txt`).
It scrapes the periodic metric lines, e.g.:

    INFO ... step:19K smpl:149K ... loss:0.164 ... l1_loss:0.144 kld_loss:0.002

and writes:
  - results/training_metrics.csv   (always)
  - results/training_curve.png     (if matplotlib is installed: `uv pip install matplotlib`)
"""
import csv
import re
import sys
from pathlib import Path

METRIC_RE = re.compile(
    r"step:(?P<step>[\d.]+)K.*?loss:(?P<loss>[\d.]+).*?l1_loss:(?P<l1>[\d.]+).*?kld_loss:(?P<kld>[\d.]+)"
)


def parse(log_path: Path):
    rows = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = METRIC_RE.search(line)
        if m:
            rows.append(
                {
                    "step": int(float(m["step"]) * 1000),
                    "loss": float(m["loss"]),
                    "l1_loss": float(m["l1"]),
                    "kld_loss": float(m["kld"]),
                }
            )
    # de-duplicate on step (rounded to K), keep last
    dedup = {r["step"]: r for r in rows}
    return [dedup[s] for s in sorted(dedup)]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    log_path = Path(sys.argv[1])
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/training_curve.png")
    rows = parse(log_path)
    if not rows:
        sys.exit("No metric lines found — is this a lerobot-train log?")

    csv_path = Path("results/training_metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "l1_loss", "kld_loss"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} points)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping PNG (`uv pip install matplotlib` to enable).")
        return

    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [r["loss"] for r in rows], label="total loss")
    ax.plot(steps, [r["l1_loss"] for r in rows], label="L1 (action) loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("ACT training loss — so101-clip-bowl")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
