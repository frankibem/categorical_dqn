import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cntk as C
from cntk.layers import Sequential, Dense

from categorical.agent import CategoricalAgent
from common.schedules import LinearSchedule
from common.replay_buffer import ReplayBuffer


def main():
    env = gym.make('CartPole-v0')
    state_shape = env.observation_space.shape
    action_count = env.action_space.n

    n = 9
    vmin, vmax = -4, 4

    model_func = Sequential([
        Dense(16, activation=C.relu),
        Dense(16, activation=C.relu),
        Dense((action_count, n))
    ])

    buffer_capacity = 128
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Fill the buffer with randomly generated samples
    state = env.reset()
    for i in range(buffer_capacity):
        action = env.action_space.sample()
        post_state, reward, done, _ = env.step(action)
        replay_buffer.add(state.astype(np.float32), action, reward, post_state.astype(np.float32), float(done))

        if done:
            state = env.reset()

    minibatch_size = 32
    max_episodes = 10000
    update_freq = 40
    log_freq = 100
    reward_buffer = np.zeros(max_episodes, dtype=np.float32)
    losses = []

    epsilon_schedule = LinearSchedule(1, 0.05, max_episodes)
    agent = CategoricalAgent(state_shape, action_count, model_func, vmin, vmax, n, lr=0.000025)

    for episode in range(1, max_episodes + 1):
        state = env.reset().astype(np.float32)
        done = False

        while not done:
            action = agent.act(state, epsilon_schedule.value(episode))
            post_state, reward, done, _ = env.step(action)

            post_state = post_state.astype(np.float32)
            replay_buffer.add(state, action, reward, post_state, float(done))
            reward_buffer[episode - 1] += reward

        minibatch = replay_buffer.sample(minibatch_size)
        agent.train(*minibatch)
        loss = agent.trainer.previous_minibatch_loss_average
        losses.append(loss)

        if episode % update_freq == 0:
            agent.update_target()

        if episode % log_freq == 0:
            average = np.sum(reward_buffer[episode - log_freq: episode]) / log_freq
            print('Episode {:4d} | Loss: {:6.4f} | Reward: {}'.format(episode, loss, average))

    agent.model.save('cartpole.cdqn')

    pd.Series(reward_buffer).rolling(window=log_freq).mean().plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    main()
