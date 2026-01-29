import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from torch.distributions import Distribution, Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, hidden_dim=256):
        super(VAE, self).__init__()
        # 编码器结构：学习数据的特征
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)  # 输入：状态+动作
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)  # 输出：均值和对数标准差
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        # 解码器结构：根据状态和隐藏向量z 表达策略动作
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)  # 输入：状态+隐藏向量z
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        # 编码过程：拼接状态和动作,并流过e1、e2
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)  # 重参数化采样技巧
        # 解码过程：
        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:  # 没有隐藏向量z(通常表示在推断或测试阶段)
            # 在测试时限制潜在空间的范围,避免生成离群的极端动作,提高生成的稳健性
            # z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)
        # 解码过程：拼接状态和隐藏向量,流过d1、d2、d3
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def sample(self, state):  # 测试阶段解码
        return self.decode(state)


class BC_CVAE(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 hidden_dim=32,
                 ):

        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim,
                       max_action, device, hidden_dim=hidden_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def sample_action(self, state):  # 动作采样
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.vae.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)  # BC环境下,模型不学习r和next_state

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)  # 生成的a',均值和方差
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss  # 动作损失,和分布误差

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()


    def save_model(self, dir):
        torch.save(self.vae.state_dict(), f'{dir}/vae.pth')

    def load_model(self, dir):
        self.vae.load_state_dict(torch.load(f'{dir}/vae.pth'))
