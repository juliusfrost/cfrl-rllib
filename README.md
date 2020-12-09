# Counterfactual RL Explanations for RLlib

## Install

```bash
pip install -r requirements.txt
```

## Run training script
```bash
python train.py --experiment minigrid-a2c-all --env MiniGrid-FourRooms-v0
```

## Run rollout script
example:
```bash
python rollout.py --run PPO --env DrivingPLE-v0 --video-dir $video_dir --episodes 10 $checkpoint_file
```
