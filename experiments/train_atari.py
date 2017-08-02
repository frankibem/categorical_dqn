import os
import gym
import numpy as np
import cntk as C
from cntk.layers import Sequential, Convolution2D, Dense

from common.atari_wrappers import wrap_env
from common.schedules import LinearSchedule
from common.replay_buffer import ReplayBuffer
from categorical.agent import CategoricalAgent


def main(env_name='KungFuMasterNoFrameskip-v0',
         train_freq=4,
         target_update_freq=10000,
         checkpoint_freq=100000,
         log_freq=1,
         batch_size=32,
         train_after=200000,
         max_timesteps=5000000,
         buffer_size=50000,
         vmin=-10,
         vmax=10,
         n=51,
         gamma=0.99,
         final_eps=0.02,
         learning_rate=0.00025,
         momentum=0.95):
    env = gym.make(env_name)
    env = wrap_env(env)

    state_dim = (4, 84, 84)
    action_count = env.action_space.n

    with C.default_options(activation=C.relu, init=C.he_uniform()):
        model_func = Sequential([
            Convolution2D((8, 8), 32, strides=4, name='conv1'),
            Convolution2D((4, 4), 64, strides=2, name='conv2'),
            Convolution2D((3, 3), 64, strides=1, name='conv3'),
            Dense(512, name='dense1'),
            Dense((action_count, n), activation=None, name='out')
        ])

    agent = CategoricalAgent(state_dim, action_count, model_func, vmin, vmax, n, gamma,
                             lr=learning_rate, mm=momentum, use_tensorboard=True)
    logger = agent.writer

    epsilon_schedule = LinearSchedule(1.0, final_eps, max_timesteps)
    replay_buffer = ReplayBuffer(buffer_size)

    try:
        obs = env.reset()
        episode = 0
        rewards = 0
        steps = 0

        for t in range(max_timesteps):
            # Take action
            if t > train_after:
                action = agent.act(obs, epsilon=epsilon_schedule.value(t))
            else:
                action = np.random.choice(action_count)
            obs_, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.add(obs, action, reward, obs_, float(done))

            obs = obs_
            rewards += reward

            if done:
                steps = t - steps + 1
                episode += 1
                obs = env.reset()

            if t > train_after and (t % train_freq) == 0:
                # Minimize error in projected Bellman update on a batch sampled from replay buffer
                experience = replay_buffer.sample(batch_size)
                agent.train(*experience)  # experience is (s, a, r, s_, t) tuple

            if t > train_after and (t % target_update_freq) == 0:
                agent.update_target()

            if done and (episode % log_freq) == 0:
                logger.write_value('rewards', rewards, episode)
                logger.write_value('steps', steps, episode)
                logger.write_value('epsilon', epsilon_schedule.value(t), episode)
                agent.trainer.summarize_training_progress()
                logger.flush()

                rewards = 0
                steps = t

            if t > train_after and (t % checkpoint_freq) == 0:
                agent.checkpoint('checkpoints/model_{}.chkpt'.format(t))

    finally:
        agent.save_model('checkpoints/{}.cdqn'.format(env_name))


if __name__ == '__main__':
    main()
