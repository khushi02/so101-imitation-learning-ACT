# Reproduce this

Exact commands used, in order. Generic arm assembly/wiring follows the official
[SO-101](https://huggingface.co/docs/lerobot/so101) and [imitation-learning](https://huggingface.co/docs/lerobot/il_robots)
guides and isn't repeated here — this documents the choices specific to *this* reproduction. Replace `<...>` and
`khushiiw/...` with your own ports and Hub username.

Every command assumes the env is active:

```bash
cd ~/Dev/lerobot && source .venv/bin/activate
```

## 1. Ports & calibration

```bash
lerobot-find-port                                    # run once per arm to get its port
lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader
```
> Pre-assembled arms ship with motor IDs already set — **skip `lerobot-setup-motors`** (it needs each motor isolated, i.e. disassembly). If calibration connects without a ping error, the IDs are fine.

## 2. Cameras

```bash
lerobot-find-cameras opencv     # note which index is the wrist cam vs the front webcam
```
Both at 640×480 @ 30 fps. The dataset keys (`wrist`, `front`) must stay consistent through record → train → eval.

## 3. Record 50 demonstrations

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" \
  --display_data=true \
  --dataset.repo_id=khushiiw/so101-clip-bowl \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the clip and place it in the bowl" \
  --dataset.private=true
```
Controls: **`n`/→** next episode · **`r`/←** re-record · **`q`/Esc** finish. Randomize the clip position each episode within a fixed region; keep the bowl, cameras, and lighting fixed.

## 4. Train ACT (Hugging Face Jobs)

Default ACT hyperparameters = the faithful reproduction. Push checkpoints to the Hub so crashes are resumable:

```bash
lerobot-train \
  --dataset.repo_id=khushiiw/so101-clip-bowl \
  --policy.type=act \
  --policy.repo_id=khushiiw/act-clip-bowl \
  --policy.device=cuda \
  --save_checkpoint_to_hub=true \
  --save_freq=5000 \
  --job.target=a10g-large
```
Resume after a crash (only works once a checkpoint has been pushed):
```bash
lerobot-train --config_path=khushiiw/act-clip-bowl --resume=true --save_checkpoint_to_hub=true --job.target=a10g-large
```
See [notes.md](notes.md) for the auto-resume loop and why `--save_checkpoint_to_hub` is essential.

## 5. Evaluate on the robot

Autonomous rollout with the trained policy (same physical setup as recording):

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=khushiiw/act-clip-bowl \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" \
  --task="Pick up the clip and place it in the bowl" \
  --duration=30 \
  --display_data=true
```
Score **success = clip ends up in the bowl** over ~15–20 trials with the clip at varied positions; log outcomes into [`../results/success_rates.csv`](../results/success_rates.csv).

## Media (for the README visuals)

- **Hero / rollout GIF:** screen-record the rerun window (or the arm) during a successful `lerobot-rollout`, convert to GIF, save as `results/media/rollout.gif`.
- **Training curve:** run `python scripts/plot_training.py <path-to-run-log> results/training_curve.png` (see the script header).
- **Dataset sample:** grab a few frames from `lerobot-dataset-viz --repo-id khushiiw/so101-clip-bowl --episode-index 0`.
