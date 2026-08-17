# Notes & gotchas

Most of the friction faced in this process was during the training process, documented below. All other slowdowns were standard componenets of the calibration process that need to be repeated if performing the experiment over multiple days.

## Training on HF Jobs

- **Jobs failed at random steps** (13%, 41%, …) with the training itself healthy each time — a sign of cloud-node
  instability (preemption / transient faults), not a bug in the run.
- **Always train with `--save_checkpoint_to_hub=true`.** Without it, checkpoints live only on the job's ephemeral
  disk and are lost when it dies — every crash means restarting from step 0. We learned this the expensive way.
- **`--save_freq=5000`** (vs the 20k default) so a crash costs ≤5k steps, and so the *first* checkpoint lands early
  (~13 min). Resume can't work until at least one checkpoint exists — and cancelling a run before that first
  checkpoint leaves nothing to resume from.
- **GPU sizing:** ACT is tiny (~3.7 GB VRAM at batch 8). A bigger *GPU* is wasted money; we used `a10g-large`
  mainly for its **RAM/CPU** headroom (video dataloading), not the accelerator.