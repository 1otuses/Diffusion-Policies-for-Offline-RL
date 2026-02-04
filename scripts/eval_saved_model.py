"""Evaluate a saved model and optionally record a video in a MuJoCo environment.

Usage examples:

# Evaluate the final saved model (actor.pth) for 5 episodes and render to screen  使用保存好的model进行评估5回合并在屏幕上渲染
python scripts/eval_saved_model.py --results_dir results/your_run --env_name walker2d-medium-v2 --episodes 5 --render

# Load model from a specific epoch and record video(s) to the results dir  指定epoch并将视频保存到结果目录
python scripts/eval_saved_model.py --results_dir results/your_run --epoch 50 --episodes 3 --record --out_dir results/your_run/videos

# Load best epoch from best_score_offline.txt  加载最佳epoch并记录视频
python scripts/eval_saved_model.py --results_dir results/your_run --best --episodes 5 --record
"""

import argparse
import json
import os
import time

import gym
import numpy as np
import torch


def find_epoch_from_best(results_dir):  # 查找最佳epoch
    for name in ("best_score_offline.txt", "best_score_online.txt"):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return int(data.get('epoch'))
    return None


def load_variant(results_dir):  # 加载variant.json文件
    vpath = os.path.join(results_dir, 'variant.json')
    if os.path.exists(vpath):
        with open(vpath, 'r') as f:
            return json.load(f)
    return {}


def build_agent_from_variant(variant, device):  # 根据variant配置构建agent
    algo = variant.get('algo', 'ql')  # 默认使用ql算法
    state_dim = variant.get('state_dim')
    action_dim = variant.get('action_dim')
    max_action = variant.get('max_action', 1.0)

    if algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=variant.get('discount', 0.99),
                      tau=variant.get('tau', 0.005),
                      beta_schedule=variant.get('beta_schedule', 'vp'),
                      n_timesteps=variant.get('T', 5),
                      lr=variant.get('lr', 2e-4))
    else:
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=variant.get('discount', 0.99),
                      tau=variant.get('tau', 0.005),
                      max_q_backup=variant.get('max_q_backup', False),
                      beta_schedule=variant.get('beta_schedule', 'vp'),
                      n_timesteps=variant.get('T', 5),
                      eta=variant.get('eta', 1.0),
                      lr=variant.get('lr', 3e-4),
                      lr_decay=variant.get('lr_decay', False),
                      lr_maxt=variant.get('num_epochs', 1),
                      grad_norm=variant.get('gn', 1.0))
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help='Experiment results directory')
    parser.add_argument('--env_name', type=str, default=None, help='Gym env name (optional, will be read from variant.json if absent)')
    parser.add_argument('--epoch', type=int, default=None, help='Load model files with this epoch id (actor_{epoch}.pth). If omitted, will load actor.pth')
    parser.add_argument('--best', action='store_true', help='Load epoch from best_score_*.txt')
    parser.add_argument('--episodes', type=int, default=5)  # 评估的回合数
    parser.add_argument('--render', action='store_true', help='Render frames to screen')  # 是否渲染
    parser.add_argument('--record', action='store_true', help='Record video(s) to disk using gym.wrappers.RecordVideo')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for videos (defaults to <results_dir>/videos)')
    parser.add_argument('--record_all', action='store_true', help='Record all episodes instead of only the last one')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    variant = load_variant(args.results_dir)  # 加载variant配置
    env_name = args.env_name or variant.get('env_name')
    if env_name is None:
        raise RuntimeError('env_name must be provided either via --env_name or variant.json in results_dir')

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'

    # Determine epoch to load
    epoch = args.epoch
    if args.best and epoch is None:
        epoch = find_epoch_from_best(args.results_dir)  # 查找最佳epoch
    # If epoch is None we will try loading actor.pth

    # Build env
    if args.record:  # 记录视频
        # RecordVideo needs gym env created normally and then wrapped
        env = gym.make(env_name)
        video_dir = args.out_dir or os.path.join(args.results_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        # By default record only the last episode to avoid saving many videos.
        if getattr(args, 'record_all', False):
            env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda e: True)
        else:
            env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda e, last=args.episodes: e == (last - 1))
    else:
        env = gym.make(env_name)

    # Read env info from variant if available
    if 'state_dim' in variant:
        state_dim = variant['state_dim']
    else:
        state_dim = env.observation_space.shape[0]
    if 'action_dim' in variant:
        action_dim = variant['action_dim']
    else:
        action_dim = env.action_space.shape[0]

    agent = build_agent_from_variant(variant, device)  # 构建agent

    # Load model files
    if epoch is not None:
        actor_path = os.path.join(args.results_dir, f'actor_{epoch}.pth')
        critic_path = os.path.join(args.results_dir, f'critic_{epoch}.pth')
    else:
        actor_path = os.path.join(args.results_dir, 'actor.pth')
        critic_path = os.path.join(args.results_dir, 'critic.pth')

    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor file not found: {actor_path}")
    agent.load_model(args.results_dir, id=epoch) if epoch is not None else agent.load_model(args.results_dir)

    returns = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        r = 0.0
        while not done:
            action = agent.sample_action(np.array(obs))
            obs, reward, done, info = env.step(action)
            r += reward
            if args.render:
                try:
                    env.render(mode='rgb_array')
                except Exception:
                    pass
            # small sleep so rendering is visible
            if args.render:
                time.sleep(0.02)
        returns.append(r)
        print(f"Episode {ep + 1}/{args.episodes} return: {r}")

    print(f"Average return: {np.mean(returns):.3f} +/- {np.std(returns):.3f}")

    if args.record:
        print(f"Videos saved to: {video_dir}")


if __name__ == '__main__':
    main()