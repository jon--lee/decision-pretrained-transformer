import numpy as np
import scipy
import torch

from ctrls.ctrl_bandit import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DarkroomOptPolicy(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.goal = env.goal

    def reset(self):
        return

    def act(self, state):
        return self.env.opt_action(state)        


class DarkroomTransformerController(Controller):
    def __init__(self, model, batch_size=1, sample=False):
        self.model = model
        self.state_dim = model.config['state_dim']
        self.action_dim = model.config['action_dim']
        self.horizon = model.horizon
        self.zeros = torch.zeros(
            batch_size, self.state_dim ** 2 + self.action_dim + 1).float().to(device)
        self.sample = sample
        self.temp = 1.0
        self.batch_size = batch_size

    def act(self, state):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(state)).float().to(device)
        if self.batch_size == 1:
            states = states[None, :]
        self.batch['query_states'] = states

        actions = self.model(self.batch).cpu().detach().numpy()
        if self.batch_size == 1:
            actions = actions[0]

        if self.sample:
            if self.batch_size > 1:
                action_indices = []
                for idx in range(self.batch_size):
                    probs = scipy.special.softmax(actions[idx] / self.temp)
                    sampled_action = np.random.choice(
                        np.arange(self.action_dim), p=probs)
                    action_indices.append(sampled_action)
            else:
                probs = scipy.special.softmax(actions / self.temp)
                action_indices = [np.random.choice(
                    np.arange(self.action_dim), p=probs)]
        else:
            action_indices = np.argmax(actions, axis=-1)

        actions = np.zeros((self.batch_size, self.action_dim))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        if self.batch_size == 1:
            actions = actions[0]
        return actions
