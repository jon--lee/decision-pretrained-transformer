import argparse
import os
import pickle
import random

import gym
import numpy as np
from skimage.transform import resize
from IPython import embed

import common_args
from envs import darkroom_env, bandit_env
from utils import (
    build_bandit_data_filename,
    build_darkroom_data_filename,
    build_miniworld_data_filename,
)


def rollin_bandit(env, cov, orig=False):
    H = env.H_context
    opt_a_index = env.opt_a_index
    xs, us, xps, rs = [], [], [], []

    exp = False
    if exp == False:
        cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
        alpha = np.ones(env.dim)
        probs = np.random.dirichlet(alpha)
        probs2 = np.zeros(env.dim)
        rand_index = np.random.choice(np.arange(env.dim))
        probs2[rand_index] = 1.0
        probs = (1 - cov) * probs + cov * probs2
    else:
        raise NotImplementedError

    for h in range(H):
        x = np.array([1])
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim), p=probs)
        u[i] = 1.0
        xp, r = env.transit(x, u)

        xs.append(x)
        us.append(u)
        xps.append(xp)
        rs.append(r)

    xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
    return xs, us, xps, rs


def rollin_mdp(env, rollin_type):
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards


def rand_pos_and_dir(env):
    pos_vec = np.random.uniform(0, env.size, size=3)
    pos_vec[1] = 0.0
    dir_vec = np.random.uniform(0, 2 * np.pi)
    return pos_vec, dir_vec


def rollin_mdp_miniworld(env, horizon, rollin_type, target_shape=(25, 25, 3)):
    observations = []
    pos_and_dirs = []
    actions = []
    rewards = []

    for _ in range(horizon):
        if rollin_type == 'uniform':
            init_pos, init_dir = rand_pos_and_dir(env)
            env.place_agent(pos=init_pos, dir=init_dir)

        obs = env.render_obs()
        obs = resize(obs, target_shape, anti_aliasing=True)
        observations.append(obs)
        pos_and_dirs.append(np.concatenate(
            [env.agent.pos[[0, -1]], env.agent.dir_vec[[0, -1]]]))

        if rollin_type == 'uniform':
            action = np.random.randint(env.action_space.n)
        elif rollin_type == 'expert':
            action = env.opt_a(obs, env.agent.pos, env.agent.dir_vec)
        else:
            raise ValueError("Invalid rollin type")
        _, rew, _, _, _ = env.step(action)
        a_zero = np.zeros(env.action_space.n)
        a_zero[action] = 1

        actions.append(a_zero)
        rewards.append(rew)

    observations = np.array(observations)
    states = np.array(pos_and_dirs)[..., 2:]    # only use dir, not pos
    actions = np.array(actions)
    rewards = np.array(rewards)
    return observations, states, actions, rewards


def generate_bandit_histories_from_envs(envs, n_hists, n_samples, cov):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin_bandit(env, cov=cov)
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'means': env.means,
                }
                trajs.append(traj)
    return trajs


def generate_mdp_histories_from_envs(envs, n_hists, n_samples, rollin_type):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin_mdp(env, rollin_type=rollin_type)
            for k in range(n_samples):
                query_state = env.sample_state()
                optimal_action = env.opt_action(query_state)

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': env.goal,
                }

                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index

                trajs.append(traj)
    return trajs


def generate_bandit_histories(n_envs, dim, horizon, var, type, **kwargs):
    envs = [bandit_env.sample(dim, horizon, var, type=type)
            for _ in range(n_envs)]
    trajs = generate_bandit_histories_from_envs(envs, **kwargs)
    return trajs


def generate_darkroom_histories(goals, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in goals]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


def generate_darkroom_permuted_histories(indices, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnvPermuted(
        dim, index, horizon) for index in indices]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


def generate_miniworld_histories(env_ids, image_dir, n_hists, n_samples, horizon, target_shape, rollin_type='uniform'):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    n_envs = len(env_ids)
    env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
    obs = env.reset()

    trajs = []
    for i, env_id in enumerate(env_ids):
        print(f"Generating histories for env {i}/{n_envs}")
        env.set_task(env_id)
        env.reset()
        for j in range(n_hists):
            (
                context_images,
                context_states,
                context_actions,
                context_rewards,
            ) = rollin_mdp_miniworld(env, horizon, rollin_type=rollin_type, target_shape=target_shape)
            filepath = f'{image_dir}/context{i}_{j}.npy'
            np.save(filepath, context_images)

            for _ in range(n_samples):
                init_pos, init_dir = rand_pos_and_dir(env)
                env.place_agent(pos=init_pos, dir=init_dir)
                obs = env.render_obs()
                obs = resize(obs, target_shape, anti_aliasing=True)

                action = env.opt_a(obs, env.agent.pos, env.agent.dir_vec)
                one_hot_action = np.zeros(env.action_space.n)
                one_hot_action[action] = 1

                traj = {
                    'query_image': obs,
                    'query_state': env.agent.dir_vec[[0, -1]],
                    'optimal_action': one_hot_action,
                    'context_images': filepath,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_states,  # unused
                    'context_rewards': context_rewards,
                    'env_id': env_id,  # not used during training, only used for evaling in correct env
                }
                trajs.append(traj)
    return trajs


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    cov = args['cov']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']


    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }

    if env == 'bandit':
        config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})

        train_trajs = generate_bandit_histories(n_train_envs, **config)
        test_trajs = generate_bandit_histories(n_test_envs, **config)
        eval_trajs = generate_bandit_histories(n_eval_envs, **config)

        train_filepath = build_bandit_data_filename(env, n_envs, config, mode=0)
        test_filepath = build_bandit_data_filename(env, n_envs, config, mode=1)
        eval_filepath = build_bandit_data_filename(env, n_eval_envs, config, mode=2)


    elif env == 'darkroom_heldout':

        config.update({'dim': dim, 'rollin_type': 'uniform'})
        goals = np.array([[(j, i) for i in range(dim)]
                         for j in range(dim)]).reshape(-1, 2)
        np.random.RandomState(seed=0).shuffle(goals)
        train_test_split = int(.8 * len(goals))
        train_goals = goals[:train_test_split]
        test_goals = goals[train_test_split:]

        eval_goals = np.array(test_goals.tolist() *
                              int(100 // len(test_goals)))
        train_goals = np.repeat(train_goals, n_envs // (dim * dim), axis=0)
        test_goals = np.repeat(test_goals, n_envs // (dim * dim), axis=0)

        train_trajs = generate_darkroom_histories(train_goals, **config)
        test_trajs = generate_darkroom_histories(test_goals, **config)
        eval_trajs = generate_darkroom_histories(eval_goals, **config)

        train_filepath = build_darkroom_data_filename(
            env, n_envs, config, mode=0)
        test_filepath = build_darkroom_data_filename(
            env, n_envs, config, mode=1)
        eval_filepath = build_darkroom_data_filename(env, 100, config, mode=2)


    elif env == 'miniworld':
        import gymnasium as gym
        import miniworld

        config.update({'rollin_type': 'uniform', 
            'target_shape': (25, 25, 3),
        })

        if env_id_start < 0 or env_id_end < 0:
            env_id_start = 0
            env_id_end = n_envs

        env_ids = np.arange(env_id_start, env_id_end)

        train_test_split = int(.8 * len(env_ids))
        train_env_ids = env_ids[:train_test_split]
        test_env_ids = env_ids[train_test_split:]

        train_filepath = build_miniworld_data_filename(
            env, env_id_start, env_id_end, config, mode=0)
        test_filepath = build_miniworld_data_filename(
            env, env_id_start, env_id_end, config, mode=1)
        eval_filepath = build_miniworld_data_filename(env, 0, 100, config, mode=2)


        train_trajs = generate_miniworld_histories(
            train_env_ids,
            train_filepath.split('.')[0],
            **config)
        test_trajs = generate_miniworld_histories(
            test_env_ids,
            test_filepath.split('.')[0],
            **config)
        eval_trajs = generate_miniworld_histories(
            test_env_ids[:100],
            eval_filepath.split('.')[0],
            **config)

    else:
        raise NotImplementedError


    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")
