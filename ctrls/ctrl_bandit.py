import itertools

import numpy as np
import scipy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Controller:
    def set_batch(self, batch):
        self.batch = batch

    def set_batch_numpy_vec(self, batch):
        self.set_batch(batch)

    def set_env(self, env):
        self.env = env


class OptPolicy(Controller):
    def __init__(self, env, batch_size=1):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        return self.env.opt_a


    def act_numpy_vec(self, x):
        opt_as = [ env.opt_a for env in self.env ]
        return np.stack(opt_as, axis=0)
        # return np.tile(self.env.opt_a, (self.batch_size, 1))


class GreedyOptPolicy(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def reset(self):
        return

    def act(self, x):
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
        i = np.argmax(rewards)
        a = self.batch['context_actions'].cpu().detach().numpy()[0][i]
        self.a = a
        return self.a


class EmpMeanPolicy(Controller):
    def __init__(self, env, online=False, batch_size = 1):
        super().__init__()
        self.env = env
        self.online = online
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        i = np.argmax(b_mean)
        j = np.argmin(counts)
        if self.online and counts[j] == 0:
            i = j
        
        a = np.zeros(self.env.dim)
        a[i] = 1.0

        self.a = a
        return self.a

    def act_numpy_vec(self, x):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        i = np.argmax(b_mean, axis=-1)
        j = np.argmin(counts, axis=-1)
        if self.online:
            mask = (counts[np.arange(self.batch_size), j] == 0)
            i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0

        self.a = a
        return self.a



class ThompsonSamplingPolicy(Controller):
    def __init__(self, env, std=.1, sample=False, prior_mean=.5, prior_var=1/12.0, warm_start=False, batch_size=1):
        super().__init__()
        self.env = env
        self.variance = std**2
        self.prior_mean = prior_mean
        self.prior_variance = prior_var
        self.batch_size = batch_size

        self.reset()
        self.sample = sample
        self.warm_start = warm_start

    def reset(self):
        if self.batch_size > 1:
            self.means = np.ones((self.batch_size, self.env.dim)) * self.prior_mean
            self.variances = np.ones((self.batch_size, self.env.dim)) * self.prior_variance
            self.counts = np.zeros((self.batch_size, self.env.dim))
        else:
            self.means = np.ones(self.env.dim) * self.prior_mean
            self.variances = np.ones(self.env.dim) * self.prior_variance
            self.counts = np.zeros(self.env.dim)

    def set_batch(self, batch):
        self.reset()
        self.batch = batch
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        for i in range(len(actions)):
            c = np.argmax(actions[i])
            self.counts[c] += 1

        for c in range(self.env.dim):
            arm_rewards = rewards[np.argmax(actions, axis=1) == c]
            self.update_posterior(c, arm_rewards)

    def set_batch_numpy_vec(self, batch):
        self.reset()
        self.batch = batch
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards'][:, :, 0]

        for i in range(len(actions[0])):
            c = np.argmax(actions[:, i], axis=-1)
            self.counts[np.arange(self.batch_size), c] += 1

        arm_means = np.zeros((self.batch_size, self.env.dim))
        for idx in range(self.batch_size):
            actions_idx = np.argmax(actions[idx], axis=-1)
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                if self.counts[idx, c] > 0:
                    arm_mean = np.mean(arm_rewards)
                    arm_means[idx, c] = arm_mean

        assert arm_means.shape[0] == self.batch_size
        assert arm_means.shape[1] == self.env.dim

        self.update_posterior_all(arm_means)

    def update_posterior(self, c, arm_rewards):
        n = self.counts[c]

        if n > 0:
            arm_mean = np.mean(arm_rewards)
            prior_weight = self.prior_variance / (self.prior_variance + (n * self.variance))
            new_mean = prior_weight * self.prior_mean + (1 - prior_weight) * arm_mean
            new_variance = 1 / (1 / self.prior_variance + n / self.variance)

            self.means[c] = new_mean
            self.variances[c] = new_variance

    def update_posterior_all(self, arm_means):
        prior_weight = self.prior_variance / (self.prior_variance + (self.counts * self.variance))
        new_mean = prior_weight * self.prior_mean + (1 - prior_weight) * arm_means
        new_variance = 1 / (1 / self.prior_variance + self.counts / self.variance)

        mask = (self.counts > 0)
        self.means[mask] = new_mean[mask]
        self.variances[mask] = new_variance[mask]

    def act(self, x):
        if self.sample:
            values = np.random.normal(self.means, np.sqrt(self.variances))
            i = np.argmax(values)

            actions = self.batch['context_actions'].cpu().detach().numpy()[0]
            rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

            if self.warm_start:
                counts = np.zeros(self.env.dim)
                for j in range(len(actions)):
                    c = np.argmax(actions[j])
                    counts[c] += 1
                j = np.argmin(counts)
                if counts[j] == 0:
                    i = j
        else:
            values = np.random.normal(self.means, np.sqrt(self.variances), size=(100, self.env.dim))
            amax = np.argmax(values, axis=1)
            freqs = np.bincount(amax, minlength=self.env.dim)
            i = np.argmax(freqs)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a

        return self.a

    def act_numpy_vec(self, x):
        if self.sample:
            values = np.random.normal(self.means, np.sqrt(self.variances))
            action_indices = np.argmax(values, axis=-1)

            actions = self.batch['context_actions']
            rewards = self.batch['context_rewards']

        else:
            values = np.stack([
                np.random.normal(self.means, np.sqrt(self.variances))
                for _ in range(100)], axis=1)
            amax = np.argmax(values, axis=-1)
            freqs = np.array([np.bincount(am, minlength=self.env.dim) for am in amax])
            action_indices = np.argmax(freqs, axis=-1)

        actions = np.zeros((self.batch_size, self.env.dim))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        self.a = actions
        return self.a



class PessMeanPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        pens = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean - pens

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a


    def act_numpy_vec(self, x):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean - bons

        i = np.argmax(bounds, axis=-1)
        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a



class UCBPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a

    def act_numpy_vec(self, x):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds, axis=-1)
        j = np.argmin(counts, axis=-1)
        mask = (counts[np.arange(200), j] == 0)
        i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a


class BanditTransformerController(Controller):
    def __init__(self, model, sample=False,  batch_size=1):
        self.model = model
        self.du = model.config['action_dim']
        self.dx = model.config['state_dim']
        self.H = model.horizon
        self.sample = sample
        self.batch_size = batch_size
        self.zeros = torch.zeros(batch_size, self.dx**2 + self.du + 1).float().to(device)

    def set_env(self, env):
        return

    def set_batch_numpy_vec(self, batch):
        # Convert each element of the batch to a torch tensor
        new_batch = {}
        for key in batch.keys():
            new_batch[key] = torch.tensor(batch[key]).float().to(device)
        self.set_batch(new_batch)

    def act(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(x)[None, :].float().to(device)
        self.batch['query_states'] = states

        a = self.model(self.batch)
        a = a.cpu().detach().numpy()[0]

        if self.sample:
            probs = scipy.special.softmax(a)
            i = np.random.choice(np.arange(self.du), p=probs)
        else:
            i = np.argmax(a)

        a = np.zeros(self.du)
        a[i] = 1.0
        return a

    def act_numpy_vec(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(x))
        if self.batch_size == 1:
            states = states[None,:]
        states = states.float().to(device)
        self.batch['query_states'] = states

        a = self.model(self.batch)
        a = a.cpu().detach().numpy()
        if self.batch_size == 1:
            a = a[0]

        if self.sample:
            probs = scipy.special.softmax(a, axis=-1)
            action_indices = np.array([np.random.choice(np.arange(self.du), p=p) for p in probs])
        else:
            action_indices = np.argmax(a, axis=-1)

        actions = np.zeros((self.batch_size, self.du))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        return actions


class LinUCB(OptPolicy):
    def __init__(self, env, k, const=1.0):
        super().__init__(env)
        self.rand = True
        self.t = 0
        self.k = k
        self.theta = np.zeros(self.env.dim)
        self.cov = 1.0 * np.eye(self.env.dim)
        self.const = const

    def act(self, x):
        if len(self.batch['context_rewards'][0]) < 1:
            hot_vector = np.zeros(self.env.dim)
            indices = np.random.choice(
                self.env.dim, size=self.env.k, replace=False)
            hot_vector[indices] = 1
            self.t += 1
            return hot_vector
        else:
            actions = self.batch['context_actions'].cpu().detach().numpy()[0]
            rewards = self.batch['context_rewards'].cpu(
            ).detach().numpy().flatten()

            cov = self.cov + actions.T @ actions
            cov_inv = np.linalg.inv(cov)

            theta = cov_inv @ actions.T @ rewards

            if self.const == 0:
                indices = np.argsort(theta)[::-1][:self.k]
                self.a = np.zeros(self.env.dim)
                self.a[indices] = 1.0
                return self.a
            else:
                k_hot_vectors = generate_k_hot_vectors(self.env.dim, self.k)
                best_arm = None
                best_value = -np.inf
                for a in k_hot_vectors:
                    value = theta @ a + self.const * np.sqrt(a @ cov_inv @ a)
                    if value > best_value:
                        best_value = value
                        best_arm = a

                self.a = best_arm
                return self.a


def generate_k_hot_vectors(n, k):
    indices = range(n)
    k_hot_vectors = []
    for combination in itertools.combinations(indices, k):
        k_hot_vector = [0] * n
        for index in combination:
            k_hot_vector[index] = 1
        k_hot_vectors.append(k_hot_vector)
    return np.array(k_hot_vectors)
