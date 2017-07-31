import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking a random number of no-ops on reset.
    No-op is assumed to be action 0
    """

    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done

        # Check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # For Qbert sometimes we stay in lives == 0 condition for a few
            # frames so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic
        and the learner need not know about any of this.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Returns only every `skip`-th frame
    """

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        """
        Repeat action, sum reward and max over last observations
        """
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        """
        Clear past frame buffer and init. to first obs. from inner env.
        """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    """
    Bin reward to {+1, 0, -1} by its sign
    """

    def _reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype(np.float32), np.array([0.299, 0.587, 0.114], np.float32))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
                                                       resample=Image.BILINEAR), dtype=np.float32)
        return frame.reshape((1, self.res, self.res))


class FrameStack(gym.Wrapper):
    """
    Buffer observations and stack across channels (last axis).
    """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """
        Clear buffer and re-fill by duplicating the first observation.
        """
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=0)


class NormalizedEnv(gym.ObservationWrapper):
    """
    Normalize the observations to [0, 1]
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def _observation(self, obs):
        return obs / 255.0


def wrap_deepmind(env, episode_life=True, clip_rewards=True):
    """
    Configure environment for DeepMind-style Atari
    Note: This does not include frame stacking
    """
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    # env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env
