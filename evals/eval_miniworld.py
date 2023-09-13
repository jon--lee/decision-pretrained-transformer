from functools import partial

import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch

import gymnasium as gym
import miniworld

from ctrls.ctrl_miniworld import (
    MiniworldOptPolicy,
    MiniworldRandPolicy,
    MiniworldTransformerController,
)
from envs.miniworld_env import MiniworldEnvVec
from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def deploy_online_vec(vec_env, controller, Heps, H, horizon, filename_template='', learner=False):
    assert H % horizon == 0

    ctx_rollouts = H // horizon

    num_envs = vec_env.num_envs
    obs_dim = (3, 25, 25)
    state_dim = 2
    action_dim = 4
    context_images = torch.zeros(
        (num_envs, ctx_rollouts, horizon, *obs_dim)).float().to(device)
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, action_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)

    cum_means = []
    for i in range(ctx_rollouts):
        batch = {
            'context_images': context_images[:, :i].reshape(num_envs, -1, *obs_dim),
            'context_states': context_states[:, :i].reshape(num_envs, -1, state_dim),
            'context_actions': context_actions[:, :i].reshape(num_envs, -1, action_dim),
            'context_rewards': context_rewards[:, :i].reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        if controller.save_video:
            controller.filename_template = partial(filename_template, ep=i)

        (
            images_lnr,
            states_lnr,
            actions_lnr,
            _,
            rewards_lnr,
        ) = vec_env.deploy_eval(controller)
        if learner:
            context_images[:, i] = images_lnr
            context_states[:, i] = torch.tensor(states_lnr)
            context_actions[:, i] = torch.tensor(actions_lnr)
            context_rewards[:, i] = torch.tensor(rewards_lnr[:, :, None])

        cum_means.append(np.sum(rewards_lnr, axis=-1))

    for h_ep in range(ctx_rollouts, Heps):
        # Reshape the batch as a singular length H = ctx_rollouts * horizon sequence.
        batch = {
            'context_images': context_images.reshape(num_envs, -1, *obs_dim),
            'context_states': context_states.reshape(num_envs, -1, state_dim),
            'context_actions': context_actions.reshape(num_envs, -1, action_dim),
            'context_rewards': context_rewards.reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        if controller.save_video:
            controller.filename_template = partial(filename_template, ep=h_ep)

        (
            images_lnr,
            states_lnr,
            actions_lnr,
            _,
            rewards_lnr,
        ) = vec_env.deploy_eval(controller)

        mean = np.sum(rewards_lnr, axis=-1)
        cum_means.append(mean)

        # Convert to torch
        images_lnr = images_lnr.float().to(device)
        states_lnr = convert_to_tensor(states_lnr)
        actions_lnr = convert_to_tensor(actions_lnr)
        rewards_lnr = convert_to_tensor(rewards_lnr[:, :, None])

        # Roll in new data by shifting the batch and appending the new data.
        if learner:
            context_images = torch.cat(
                (context_images[:, 1:], images_lnr[:, None]), dim=1)
            context_states = torch.cat(
                (context_states[:, 1:], states_lnr[:, None]), dim=1)
            context_actions = torch.cat(
                (context_actions[:, 1:], actions_lnr[:, None]), dim=1)
            context_rewards = torch.cat(
                (context_rewards[:, 1:], rewards_lnr[:, None]), dim=1)

    return np.stack(cum_means, axis=1)


def online(eval_trajs, model, Heps, horizon, H, n_eval, save_video=False, filename_template=''):
    assert H % horizon == 0

    all_means_lnr = []

    envs = []

    for i_eval in range(n_eval):
        print(f"Eval traj: {i_eval}")
        env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
        env.set_task(env_id=8000 + i_eval)
        envs.append(env)

    vec_env = MiniworldEnvVec(envs)

    # Learner
    print("Evaluating learner")
    lnr_filename_template = partial(filename_template.format, controller='lnr')
    lnr_controller = MiniworldTransformerController(
        model,
        batch_size=n_eval,
        sample=True,
        save_video=save_video,
        filename_template=lnr_filename_template)
    cum_means_lnr = deploy_online_vec(
        vec_env, lnr_controller, Heps, H, horizon, lnr_filename_template, learner=True)

    all_means_lnr = np.array(cum_means_lnr)
    means_lnr = np.mean(all_means_lnr, axis=0)
    sems_lnr = scipy.stats.sem(all_means_lnr, axis=0)

    # Optimal policy
    print("Evaluating optimal policy")
    opt_filename_template = partial(filename_template.format, controller='opt')
    opt_controller = MiniworldOptPolicy(
        vec_env, batch_size=n_eval, save_video=save_video, filename_template=opt_filename_template)
    cum_means_opt = deploy_online_vec(
        vec_env, opt_controller, 1, H, horizon, opt_filename_template)
    cum_means_opt = np.repeat(cum_means_opt, Heps, axis=-1)

    all_means_opt = np.array(cum_means_opt)
    means_opt = np.mean(all_means_opt, axis=0)
    sems_opt = scipy.stats.sem(all_means_opt, axis=0)

    # Random policy
    print("Evaluating random policy")
    rand_controller = MiniworldRandPolicy(vec_env, batch_size=n_eval)
    cum_means_rand = deploy_online_vec(
        vec_env, rand_controller, Heps, H, horizon)

    all_means_rand = np.array(cum_means_rand)
    means_rand = np.mean(all_means_rand, axis=0)
    sems_rand = scipy.stats.sem(all_means_rand, axis=0)

    # plot individual curves
    for i in range(n_eval):
        plt.plot(all_means_lnr[i], color='blue', alpha=0.2)
        plt.plot(all_means_opt[i], color='green', alpha=0.2)
        plt.plot(all_means_rand[i], color='orange', alpha=0.2)

    # plot the results with fill between
    plt.plot(means_lnr, color='blue', label='LNR')
    plt.fill_between(np.arange(Heps), means_lnr - sems_lnr,
                     means_lnr + sems_lnr, color='blue', alpha=0.2)

    plt.plot(means_opt, color='green', label='Optimal')
    plt.fill_between(np.arange(Heps), means_opt - sems_opt,
                     means_opt + sems_opt, color='green', alpha=0.2)

    plt.plot(means_rand, color='orange', label='Rand')
    plt.fill_between(np.arange(Heps), means_rand - sems_rand,
                     means_rand + sems_rand, color='orange', alpha=0.2)

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Average Reward')
    plt.title(f'Online Evaluation on {n_eval} envs')

    baselines = {
        'lnr': all_means_lnr,
        'opt': all_means_opt,
        'rand': all_means_rand,
    }
    return baselines


def offline(eval_trajs, model, n_eval, save_video=False, filename_template=''):
    all_rs_lnr = []
    all_rs_lnr_greedy = []

    envs = []
    trajs = []

    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        trajs.append(traj)
        env_id = traj['env_id']

        print(f"Eval traj id: {env_id}")

        env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
        env.set_task(env_id=env_id)
        envs.append(env)


    print("Running darkroom offline evaluations in parallel")
    vec_env = MiniworldEnvVec(envs)
    lnr_filename_template = partial(filename_template.format, controller='lnr')
    lnr = MiniworldTransformerController(
        model,
        batch_size=n_eval,
        sample=True,
        save_video=save_video,
        filename_template=lnr_filename_template)
    lnr_greedy_filename_template = partial(
        filename_template.format, controller='lnr_greedy')
    lnr_greedy = MiniworldTransformerController(
        model,
        batch_size=n_eval,
        sample=False,
        save_video=save_video,
        filename_template=lnr_greedy_filename_template)
    opt_filename_template = partial(filename_template.format, controller='opt')
    opt = MiniworldOptPolicy(
        vec_env, batch_size=n_eval, save_video=False, filename_template=opt_filename_template)
    rand = MiniworldRandPolicy(vec_env, batch_size=n_eval)

    context_images = []
    for traj in trajs:
        images = np.load(traj['context_images'])
        images = [lnr.transform(image) for image in images]
        images = torch.stack(images).float().to(device)
        context_images.append(images)
    batch = {
        'context_images': torch.stack(context_images),
        'context_states': convert_to_tensor([traj['context_states'] for traj in trajs]),
        'context_actions': convert_to_tensor([traj['context_actions'] for traj in trajs]),
        'context_rewards': convert_to_tensor([traj['context_rewards'][:, None] for traj in trajs]),
    }

    lnr.set_batch(batch)
    lnr_greedy.set_batch(batch)
    opt.set_batch(batch)
    rand.set_batch(batch)

    _, _, _, _, rs_lnr = vec_env.deploy_eval(lnr)
    _, _, _, _, rs_lnr_greedy = vec_env.deploy_eval(lnr_greedy)
    _, _, _, _, rs_opt = vec_env.deploy_eval(opt)
    _, _, _, _, rs_rand = vec_env.deploy_eval(rand)

    all_rs_lnr = np.sum(rs_lnr, axis=-1)
    all_rs_lnr_greedy = np.sum(rs_lnr_greedy, axis=-1)
    all_rs_opt = np.sum(rs_opt, axis=-1)
    all_rs_rand = np.sum(rs_rand, axis=-1)

    baselines = {
        'opt': np.array(all_rs_opt),
        'lnr': np.array(all_rs_lnr),
        'lnr_greedy': np.array(all_rs_lnr_greedy),
        'rand': np.array(all_rs_rand),
    }
    baselines_means = {
        k: np.mean(v) for k, v in baselines.items()
    }
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')
