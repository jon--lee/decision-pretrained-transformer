import gym
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseEnv(gym.Env):
    def reset(self):
        raise NotImplementedError

    def transit(self, state, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        pass

    def deploy_eval(self, ctrl):
        return self.deploy(ctrl)

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

            rews.append(rew)
            next_obs.append(ob)

        obs = np.array(obs)
        acts = np.array(acts)
        next_obs = np.array(next_obs)
        rews = np.array(rews)

        return obs, acts, next_obs, rews
