import os
import gym
import json
import numpy as np
import cntk as C


def get_action(model, state, z):
    probs = model.eval(state.astype(np.float32))[0]
    action_values = np.dot(probs, z)
    return np.argmax(action_values)


def main():
    with open('cartpole.json', encoding='utf-8') as config_file:
        config = json.load(config_file)

    if not os.path.exists('cartpole.cdqn'):
        print('Run `python -m experiments.train_cartpole` to train and save a model for CartPole-v0')
        return

    env = gym.make('CartPole-v0')

    model = C.load_model('cartpole.cdqn')
    z = np.linspace(config['vmin'], config['vmax'], config['n'], dtype=np.float32)

    rewards = 0.0
    s = env.reset()
    done = False

    while not done:
        env.render()
        a = get_action(model, s, z)
        s, r, done, info = env.step(a)

        rewards += r
    print('Total reward: {}'.format(rewards))


if __name__ == '__main__':
    main()
