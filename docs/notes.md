# Notes & gotchas

Real friction hit during this reproduction, and how to avoid it. Most of these cost hours; none are in the
official tutorial.

## Hardware / calibration

- **Pre-assembled arms:** don't run `lerobot-setup-motors`. It requires connecting one motor at a time (isolated
  from the daisy-chain) — impossible without disassembly. Vendors pre-set the IDs; go straight to calibration. A
  clean calibrate with no ping error confirms the IDs are good.
- **Leader gripper calibration:** during the joint-sweep step you must fully squeeze *and* release the leader's
  trigger. Miss it and the gripper's calibrated range is ~2 counts (vs ~1200 normal), and gripper teleop won't
  work. Check the printed `MIN`/`MAX` per joint — a collapsed range means re-calibrate.
- **Camera indices shift** on replug/reboot. Re-run `lerobot-find-cameras opencv` before recording/eval and make
  sure the `wrist` / `front` keys map to the right physical camera.
- **macOS `objc[...] Class AVFFrameReceiver is implemented in both ...`** at startup is a cosmetic `cv2` vs `av`
  warning — harmless, ignore it.

## Dataset on the Hub

- **LeRobot auto-appends a timestamp** to the local dataset folder (e.g. `so101-clip-bowl_20260809_141106`). The
  Hub repo name can differ — we uploaded the local folder to a clean name with `hf upload`.
- **Manual `hf upload` skips the codebase-version git tag** that LeRobot's own `push_to_hub` creates. Without it,
  training fails with `Your dataset must be tagged with a codebase version`. Fix:
  ```python
  from huggingface_hub import HfApi
  HfApi().create_tag("khushiiw/so101-clip-bowl", tag="v3.0", repo_type="dataset")  # tag = codebase_version in info.json
  ```

## Training on HF Jobs (the big one)

- **Jobs failed at random steps** (13%, 41%, …) with the training itself healthy each time — a sign of cloud-node
  instability (preemption / transient faults), not a bug in the run.
- **Always train with `--save_checkpoint_to_hub=true`.** Without it, checkpoints live only on the job's ephemeral
  disk and are lost when it dies — every crash means restarting from step 0. We learned this the expensive way.
- **`--save_freq=5000`** (vs the 20k default) so a crash costs ≤5k steps, and so the *first* checkpoint lands early
  (~13 min). Resume can't work until at least one checkpoint exists — and cancelling a run before that first
  checkpoint leaves nothing to resume from.
- **Auto-resume loop** (runs fresh until a checkpoint exists, then resumes on every crash):
  ```bash
  while true; do
    if python -c "import sys;from huggingface_hub import HfApi;sys.exit(0 if any(f.startswith('checkpoints/') for f in HfApi().list_repo_files('khushiiw/act-clip-bowl')) else 1)" 2>/dev/null; then
      lerobot-train --config_path=khushiiw/act-clip-bowl --resume=true --save_checkpoint_to_hub=true --job.target=a10g-large && break
    else
      lerobot-train --dataset.repo_id=khushiiw/so101-clip-bowl --policy.type=act --policy.repo_id=khushiiw/act-clip-bowl --policy.device=cuda --save_checkpoint_to_hub=true --save_freq=5000 --job.target=a10g-large && break
    fi
    echo "run ended — retrying in 30s..."; sleep 30
  done
  ```
- **GPU sizing:** ACT is tiny (~3.7 GB VRAM at batch 8). A bigger *GPU* is wasted money; we used `a10g-large`
  mainly for its **RAM/CPU** headroom (video dataloading), not the accelerator.

## Interpreting results

- **Training loss ≠ task success.** L1 loss plateaued nicely, but that says little about real success rate — the
  on-robot rollout is the only metric that counts.
