# Counterfactual RL Explanations for RLlib

## Install

```bash
pip install -r requirements.txt
```

## Run training script
```bash
python train.py
```

```
usage: train.py [-h] [--env ENV] [--config CONFIG] [--algo ALGO]
                [--framework {torch,tf,tfe}] [--suite SUITE]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment name
                        (default: PongNoFrameskip-v4)
  --config CONFIG       Config file to load algorithm from. Defaults to
                        algorithm argument choice.
                        (default: None)
  --algo ALGO           Choose algorithm from those implemented. Used if
                        config argument not set.
                        (default: ppo)
  --framework {torch,tf,tfe}
                        (default: torch)
  --suite SUITE         used for config location
                        (default: atari)
```
