import gym
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cntk as C
from cntk.layers import Sequential, Dense

from categorical.agent import CategoricalAgent
from common.schedules import LinearSchedule
from common.replay_buffer import ReplayBuffer


def main():
    with open('cartpole.json', encoding='utf-8') as config_file:
        config = json.load(config_file)

    env = gym.make('CartPole-v0')
    state_shape = env.observation_space.shape
    action_count = env.action_space.n

    layers = []
    for layer in config['layers']:
        layers.append(Dense(layer, activation=C.relu))

    layers.append(Dense((action_count, config['n']), activation=None))
    model_func = Sequential(layers)

    replay_buffer = ReplayBuffer(config['buffer_capacity'])

    # Fill the buffer with randomly generated samples
    state = env.reset()
    for i in range(config['buffer_capacity']):
        action = env.action_space.sample()
        post_state, reward, done, _ = env.step(action)
        replay_buffer.add(state.astype(np.float32), action, reward, post_state.astype(np.float32), float(done))

        if done:
            state = env.reset()

    reward_buffer = np.zeros(config['max_episodes'], dtype=np.float32)
    losses = []

    epsilon_schedule = LinearSchedule(1, 0.01, config['max_episodes'])
    agent = CategoricalAgent(state_shape, action_count, model_func, config['vmin'], config['vmax'], config['n'],
                             lr=config['lr'], gamma=config['gamma'])

    log_freq = config['log_freq']
    for episode in range(1, config['max_episodes'] + 1):
        state = env.reset().astype(np.float32)
        done = False

        while not done:
            action = agent.act(state, epsilon_schedule.value(episode))
            post_state, reward, done, _ = env.step(action)

            post_state = post_state.astype(np.float32)
            replay_buffer.add(state, action, reward, post_state, float(done))
            reward_buffer[episode - 1] += reward

            state = post_state

        minibatch = replay_buffer.sample(config['minibatch_size'])
        agent.train(*minibatch)
        loss = agent.trainer.previous_minibatch_loss_average
        losses.append(loss)

        if episode % config['target_update_freq'] == 0:
            agent.update_target()

        if episode % log_freq == 0:
            average = np.sum(reward_buffer[episode - log_freq: episode]) / log_freq
            print('Episode {:4d} | Loss: {:6.4f} | Reward: {}'.format(episode, loss, average))

    agent.model.save('cartpole.cdqn')

    sns.set_style('dark')
    pd.Series(reward_buffer).rolling(window=log_freq).mean().plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CartPole - Reward with Time')
    plt.show()

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('CartPole - Loss with Time')
    plt.show()


if __name__ == '__main__':
    main()
