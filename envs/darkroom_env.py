import itertools

import gym
import numpy as np
import torch

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DarkroomEnv(BaseEnv):
    def __init__(self, dim, goal, horizon):
        self.dim = dim
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(self.state_dim,))
        self.action_space = gym.spaces.Discrete(self.action_dim)

    def sample_state(self):
        return np.random.randint(0, self.dim, 2)

    def sample_action(self):
        i = np.random.randint(0, 5)
        a = np.zeros(self.action_space.n)
        a[i] = 1
        return a

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0])
        return self.state

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_space.n)
        state = np.array(state)
        if action == 0:
            state[0] += 1
        elif action == 1:
            state[0] -= 1
        elif action == 2:
            state[1] += 1
        elif action == 3:
            state[1] -= 1
        state = np.clip(state, 0, self.dim - 1)

        if np.all(state == self.goal):
            reward = 1
        else:
            reward = 0
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        if state[0] < self.goal[0]:
            action = 0
        elif state[0] > self.goal[0]:
            action = 1
        elif state[1] < self.goal[1]:
            action = 2
        elif state[1] > self.goal[1]:
            action = 3
        else:
            action = 4
        zeros = np.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros


class DarkroomEnvPermuted(DarkroomEnv):
    """
    Darkroom environment with permuted actions. The goal is always the bottom right corner.
    """

    def __init__(self, dim, perm_index, H):
        goal = np.array([dim - 1, dim - 1])
        super().__init__(dim, goal, H)

        self.perm_index = perm_index
        assert perm_index < 120     # 5! permutations in darkroom
        actions = np.arange(self.action_space.n)
        permutations = list(itertools.permutations(actions))
        self.perm = permutations[perm_index]

    def transit(self, state, action):
        perm_action = np.zeros(self.action_space.n)
        perm_action[self.perm[np.argmax(action)]] = 1
        return super().transit(state, perm_action)

    def opt_action(self, state):
        action = super().opt_action(state)
        action = np.argmax(action)
        perm_action = np.where(self.perm == action)[0][0]
        zeros = np.zeros(self.action_space.n)
        zeros[perm_action] = 1
        return zeros


class DarkroomEnvVec(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            done = all(done)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)
        return obs, acts, next_obs, rews
