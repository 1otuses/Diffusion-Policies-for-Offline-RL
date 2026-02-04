# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import os
import torch
import json
from tqdm import tqdm, trange

# mujoco_bin_path = r"C:\Users\Administrator\.mujoco\mujoco210\bin"  # Windows下添加MuJoCo的bin目录
# if os.path.exists(mujoco_bin_path):
#     os.add_dll_directory(mujoco_bin_path)
#     print(f"Added DLL directory: {mujoco_bin_path}")
# else:
#     print(f"WARNING: Path not found: {mujoco_bin_path}")

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
# from torch.utils.tensorboard import SummaryWriter

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 10, 'num_epochs': 50, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 10, 'num_epochs': 50, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}

def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)  # 获取数据
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)  # 创建数据采样器
    utils.print_banner('Loaded buffer')

    if args.algo == 'ql':  # 采用QL方法
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'bc':  # 纯BC方法
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)

    early_stop = False  # 是否提前停止训练
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)  # 提前停止判定
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0  # 训练步数
    # num_epochs 表示训练批次
    max_timesteps = args.num_epochs * args.num_steps_per_epoch  # 最大训练步数
    metric = 100.  
    utils.print_banner(f"Training Start", separator="*", num_star=90)  #  开始训练

    with tqdm(total=max_timesteps, desc="Training", unit="step", dynamic_ncols=True) as pbar:
        while (training_iters < max_timesteps) and (not early_stop):
            # eval_freq: 每多少个epoch评估一次
            chunks_per_eval = args.eval_freq  # epoch分割成若干评估块
            steps_per_chunk = args.num_steps_per_epoch  # 每个评估块的步数

            chunk_loss_history = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
            # 保存每个chunk的loss
            for _ in range(chunks_per_eval):
                # 每次只训练 1 个 epoch (1000步)
                loss_metric = agent.train(data_sampler,
                                      iterations=steps_per_chunk,
                                      batch_size=args.batch_size,
                                      log_writer=writer)
            # training_iters += iterations
                training_iters += steps_per_chunk
                pbar.update(steps_per_chunk)

                for k in chunk_loss_history.keys():
                    if k in loss_metric:
                        chunk_loss_history[k].append(np.mean(loss_metric[k]))

                pbar.set_postfix({  # 更新进度条
                    'Epoch': int(training_iters // args.num_steps_per_epoch),
                    'BC': f"{np.mean(loss_metric.get('bc_loss', 0)):.3f}",
                    'QL': f"{np.mean(loss_metric.get('ql_loss', 0)):.3f}",
                    'Actor': f"{np.mean(loss_metric.get('actor_loss', 0)):.3f}",
                    'Critic': f"{np.mean(loss_metric.get('critic_loss', 0)):.3f}",
                })
            # loss_metric = agent.train(data_sampler,
            #                           iterations=iterations,
            #                           batch_size=args.batch_size,
            #                           log_writer=writer)
            # training_iters += iterations
            curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

            # 计算平均loss
            avg_losses = {k: np.mean(v) if v else 0.0 for k, v in chunk_loss_history.items()}

            # avg_bc_loss = np.mean(chunk_loss_history['bc_loss'])
            # avg_ql_loss = np.mean(chunk_loss_history['ql_loss'])
            # avg_actor_loss = np.mean(chunk_loss_history['actor_loss'])
            # avg_critic_loss = np.mean(chunk_loss_history['critic_loss'])

            tqdm.write(f"Train step: {training_iters} | Epoch: {curr_epoch} | BC: {avg_losses['bc_loss']:.4f} | QL: {avg_losses['ql_loss']:.4f}")
            # Logging  日志记录每 评估块
            utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
            logger.record_tabular('Trained Epochs', curr_epoch)
            logger.record_tabular('BC Loss', avg_losses['bc_loss'])
            logger.record_tabular('QL Loss', avg_losses['ql_loss'])
            logger.record_tabular('Actor Loss', avg_losses['actor_loss'])
            logger.record_tabular('Critic Loss', avg_losses['critic_loss'])
            logger.dump_tabular()

            # Evaluation
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes)
            # eval_res: 平均回报, eval_res_std: 回报标准差, eval_norm_res: 平均归一化回报, eval_norm_res_std: 归一化回报标准差
            # 保存平均Loss而不是每次的Loss
            evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            avg_losses['bc_loss'], avg_losses['ql_loss'],
                            avg_losses['actor_loss'], avg_losses['critic_loss'],
                            curr_epoch])
            np.save(os.path.join(output_dir, "eval"), evaluations)  # 保存评估数据
            logger.record_tabular('Average Episodic Reward', eval_res)
            logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
            logger.dump_tabular()

            bc_loss = np.mean(loss_metric['bc_loss'])
            if args.early_stop:
                early_stop = stop_check(metric, bc_loss)

            metric = bc_loss

            # Model saving strategy: only save best or top-k if requested
            if args.save_best_model:
                if args.ms == 'online':
                    if not hasattr(train_agent, '_online_best_score'):
                        train_agent._online_best_score = -float('inf')  # 初始化最佳分数负无穷
                        train_agent._online_best_epoch = -1  # 初始化最佳epoch
                    if eval_norm_res > train_agent._online_best_score:  # 判断当前分数
                        if train_agent._online_best_epoch != -1:
                            for name in [f'actor_{train_agent._online_best_epoch}.pth', f'critic_{train_agent._online_best_epoch}.pth']:
                                path = os.path.join(output_dir, name)
                                try:
                                    if os.path.exists(path): os.remove(path)  # 用当前最佳模型替换之前的最佳模型
                                except Exception: pass

                        train_agent._online_best_score = eval_norm_res
                        train_agent._online_best_epoch = curr_epoch

                        agent.save_model(output_dir, curr_epoch)

                elif args.ms == 'offline':
                    current_loss = bc_loss
                    save_limit = args.top_k + 1  # 多保存一个以便比较
                    if not hasattr(train_agent, '_offline_top_k_losses'):
                        train_agent._offline_top_k_losses = [] # 存储格式: (loss, epoch)
                    
                    found = False
                    for i, (l, eid) in enumerate(train_agent._offline_top_k_losses):
                        if eid == curr_epoch:
                            train_agent._offline_top_k_losses[i] = (current_loss, curr_epoch)
                            found = True
                            break
                    if not found:
                        train_agent._offline_top_k_losses.append((current_loss, curr_epoch))

                    train_agent._offline_top_k_losses.sort(key=lambda x: x[0])
                    top_k_candidates = train_agent._offline_top_k_losses[:save_limit]  # 取前k个最小loss

                    if any(eid == curr_epoch for _, eid in top_k_candidates):
                        agent.save_model(output_dir, curr_epoch)

                    # 4. 清理掉跌出 Top-K 的废弃模型
                    while len(train_agent._offline_top_k_losses) > save_limit:
                        bad_loss, bad_epoch = train_agent._offline_top_k_losses.pop() # 弹出列表末尾(Loss最大的)
                        for name in [f'actor_{bad_epoch}.pth', f'critic_{bad_epoch}.pth']:
                            path = os.path.join(output_dir, name)
                            try:
                                if os.path.exists(path): os.remove(path)
                            except OSError: pass

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':  # 在线:选择评估分数最高的模型
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':  # 离线:选择BC损失较小的模型
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    # writer.close()


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)  # 使用不同的随机种子进行评估

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="halfcheetah-medium-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')    # action表示布尔值参数
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    # parser.add_argument("--top_k", default=1, type=int)

    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--eta", default=1.0, type=float)
    # parser.add_argument("--max_q_backup", action='store_true')
    # parser.add_argument("--reward_tune", default='no', type=str)
    # parser.add_argument("--gn", default=-1.0, type=float)

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"  # 将符号|改为_  |为Linux可行
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    # top_k: 用于离线模型选择时选择前k个模型
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    env = gym.make(args.env_name)

    # env.seed(args.seed)
    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)  # 设置日志记录器
    # 保存的路径：results/{env_name}|{exp}|diffusion-{algo}|T-{T}|ms-{ms}|k-{top_k}|seed/logger_data.txt
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
