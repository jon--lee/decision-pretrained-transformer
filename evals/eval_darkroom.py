import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from ctrls.ctrl_darkroom import (
    DarkroomOptPolicy,
    DarkroomTransformerController,
)
from envs.darkroom_env import (
    DarkroomEnv,
    DarkroomEnvPermuted,
    DarkroomEnvVec,
)
from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deploy_online_vec(vec_env, controller, Heps, H, horizon):
    assert H % horizon == 0

    ctx_rollouts = H // horizon

    num_envs = vec_env.num_envs
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.action_dim)).float().to(device)
    context_next_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)

    cum_means = []
    for i in range(ctx_rollouts):
        batch = {
            'context_states': context_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions[:, :i, :].reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards[:, :i, :, :].reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)
        context_states[:, i, :, :] = convert_to_tensor(states_lnr)
        context_actions[:, i, :, :] = convert_to_tensor(actions_lnr)
        context_next_states[:, i, :, :] = convert_to_tensor(next_states_lnr)
        context_rewards[:, i, :, :] = convert_to_tensor(rewards_lnr[:, :, None])

        cum_means.append(np.sum(rewards_lnr, axis=-1))

    for _ in range(ctx_rollouts, Heps):
        # Reshape the batch as a singular length H = ctx_rollouts * horizon sequence.
        batch = {
            'context_states': context_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions.reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards.reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)

        mean = np.sum(rewards_lnr, axis=-1)
        cum_means.append(mean)

        # Convert to torch
        states_lnr = convert_to_tensor(states_lnr)
        actions_lnr = convert_to_tensor(actions_lnr)
        next_states_lnr = convert_to_tensor(next_states_lnr)
        rewards_lnr = convert_to_tensor(rewards_lnr[:, :, None])

        # Roll in new data by shifting the batch and appending the new data.
        context_states = torch.cat(
            (context_states[:, 1:, :, :], states_lnr[:, None, :, :]), dim=1)
        context_actions = torch.cat(
            (context_actions[:, 1:, :, :], actions_lnr[:, None, :, :]), dim=1)
        context_next_states = torch.cat(
            (context_next_states[:, 1:, :, :], next_states_lnr[:, None, :, :]), dim=1)
        context_rewards = torch.cat(
            (context_rewards[:, 1:, :, :], rewards_lnr[:, None, :, :]), dim=1)

    return np.stack(cum_means, axis=1)


def online(eval_trajs, model, Heps, H, n_eval, dim, horizon, permuted=False):
    assert H % horizon == 0

    all_means_lnr = []

    envs = []
    for i_eval in range(n_eval):
        print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        if permuted:
            env = DarkroomEnvPermuted(dim, traj['perm_index'], horizon)
        else:
            env = DarkroomEnv(dim, traj['goal'], horizon)
        envs.append(env)

    lnr_controller = DarkroomTransformerController(
        model, batch_size=n_eval, sample=True)
    vec_env = DarkroomEnvVec(envs)
    cum_means_lnr = deploy_online_vec(vec_env, lnr_controller, Heps, H, horizon)

    all_means_lnr = np.array(cum_means_lnr)
    means_lnr = np.mean(all_means_lnr, axis=0)
    sems_lnr = scipy.stats.sem(all_means_lnr, axis=0)

    # Plotting
    for i in range(n_eval):
        plt.plot(all_means_lnr[i], color='blue', alpha=0.2)

    plt.plot(means_lnr, label='Learner')
    plt.fill_between(np.arange(Heps), means_lnr - sems_lnr,
                     means_lnr + sems_lnr, alpha=0.2)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title(f'Online Evaluation on {n_eval} Envs')


def offline(eval_trajs, model, n_eval, H, dim, permuted=False):
    all_rs_opt = []
    all_rs_lnr = []
    all_rs_lnr_greedy = []

    envs = []
    trajs = []

    for i_eval in range(n_eval):
        print(f"Eval traj: {i_eval}")

        traj = eval_trajs[i_eval]
        batch = {
            'context_states': convert_to_tensor(traj['context_states'][None, :, :]),
            'context_actions': convert_to_tensor(traj['context_actions'][None, :, :]),
            'context_next_states': convert_to_tensor(traj['context_next_states'][None, :, :]),
            'context_rewards': convert_to_tensor(traj['context_rewards'][None, :, None]),
        }

        if permuted:
            env = DarkroomEnvPermuted(dim, traj['perm_index'], H)
        else:
            env = DarkroomEnv(dim, traj['goal'], H)

        true_opt = DarkroomOptPolicy(env)
        true_opt.set_batch(batch)

        _, _, _, rs_opt = env.deploy_eval(true_opt)
        all_rs_opt.append(np.sum(rs_opt))

        envs.append(env)
        trajs.append(traj)

        

    print("Running darkroom offline evaluations in parallel")
    vec_env = DarkroomEnvVec(envs)
    lnr = DarkroomTransformerController(
        model, batch_size=n_eval, sample=True)
    lnr_greedy = DarkroomTransformerController(
        model, batch_size=n_eval, sample=False)

    batch = {
        'context_states': convert_to_tensor([traj['context_states'] for traj in trajs]),
        'context_actions': convert_to_tensor([traj['context_actions'] for traj in trajs]),
        'context_next_states': convert_to_tensor([traj['context_next_states'] for traj in trajs]),
        'context_rewards': convert_to_tensor([traj['context_rewards'][:, None] for traj in trajs]),
    }
    lnr.set_batch(batch)
    lnr_greedy.set_batch(batch)

    _, _, _, rs_lnr = vec_env.deploy_eval(lnr)
    _, _, _, rs_lnr_greedy = vec_env.deploy_eval(lnr_greedy)
    all_rs_lnr = np.sum(rs_lnr, axis=-1)
    all_rs_lnr_greedy = np.sum(rs_lnr_greedy, axis=-1)

    baselines = {
        'Opt': np.array(all_rs_opt),
        'Learner': np.array(all_rs_lnr),
        'Learner (greedy)': np.array(all_rs_lnr_greedy)
    }
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.ylabel('Average Return')
    plt.title(f'Average Return on {n_eval} Trajectories')
