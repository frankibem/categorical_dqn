import gym
import argparse
import numpy as np

from common.atari_wrappers import MaxAndSkipEnv, WarpFrame, NormalizedEnv, FrameStack
import cntk as C


def wrap_env(env):
    """
    Wrap the environment in a similar fashion as in training but without clipped rewards,
    and loss of life as end of an episode
    :param env: The OpenAI Gym environment to wrap
    :return: The wrapped environment
    """
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = NormalizedEnv(env)
    env = FrameStack(env, 4)

    return env


def get_action(model, state, z):
    probs = model.eval(state.astype(np.float32))[0]
    action_values = np.dot(probs, z)
    return np.argmax(action_values)


def main(env_name, model_path, vmin=-10, vmax=10, n=51):
    env = gym.make(env_name)
    env = wrap_env(env)

    z = np.linspace(vmin, vmax, n, dtype=np.float32)
    model = C.load_model(model_path)

    rewards = 0.0
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = get_action(model, state, z)
        state, r, done, _ = env.step(action)

        rewards += r

    print('Total rewards: {}'.format(rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='The name of the Atari OpenAI Gym environment')
    parser.add_argument('model', help='The path to the trained model')

    args = parser.parse_args()

    main(args.env, args.model)
