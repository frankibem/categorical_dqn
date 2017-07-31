# Categorical DQN
Attempt at CNTK implementation of Categorical DQN from 'A distributional Perspective on Reinforcement Learning' found [here](https://arxiv.org/pdf/1707.06887.pdf).

## Dependencies
1. Python 3
2. CNTK v2

## Status
Work in progress.

Current experiments show **decreasing** rewards with time :cry:. Help proofreading and hunting down bugs will be appreciated.

To run the CartPole experiment from OpenAI Gym, use:

```
python -m experiments.train_cartpole
```