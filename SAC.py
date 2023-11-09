import torch
import torch.nn as nn
from math import pi as pi_constant
from typing import Tuple

import numpy as np
import torch
import gym

import datetime
import copy
Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

from collections import namedtuple
from collections import deque
import torch
import numpy.random as nr
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import matplotlib.pyplot as plt

Transitions = namedtuple('Transitions', ['obs', 'action', 'reward', 'next_obs', 'done'])


class ReplayBuffer:
    def __init__(self, config):
        replay_buffer_size = config['replay_buffer_size']
        seed = config['seed']
        nr.seed(seed)

        self.replay_buffer_size = replay_buffer_size
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

    def append_memory(self,
                      obs,
                      action,
                      reward,
                      next_obs,
                      done: bool):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size):
        buffer_size = len(self.obs)

        idx = nr.choice(buffer_size,
                        size=min(buffer_size, batch_size),
                        replace=False)
        t = Transitions
        t.obs = torch.stack(list(map(self.obs.__getitem__, idx)))
        t.action = torch.stack(list(map(self.action.__getitem__, idx)))
        t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
        t.next_obs = torch.stack(list(map(self.next_obs.__getitem__, idx)))
        t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
        return t

    def clear(self):
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)



class SAC:
    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.smooth = config['smooth']  # smoothing coefficient for target net
        self.discount = config['discount']  # discount factor
        self.alpha = config['alpha']  # temperature parameter in SAC
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.actor = ActorNet(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1_tar = QCriticNet(dim_obs=self.dim_obs,
                                 dim_action=self.dim_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2_tar = QCriticNet(dim_obs=self.dim_obs,
                                 dim_action=self.dim_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.optimizer_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)

    def update(self, buffer):
        # sample from replay memory
        t = buffer.sample(self.batch_size)

        # update critic
        with torch.no_grad():
            next_action_sample, next_logProb_sample, next_mu_sample = self.actor(t.next_obs)
            a = self.Q1_tar(t.next_obs, next_action_sample)
            b = self.Q2_tar(t.next_obs, next_action_sample)
            Qp = torch.min(a, b)
            Q_target = t.reward + self.discount * (~t.done) * (Qp - self.alpha * next_logProb_sample)

        loss_Q1 = torch.mean((self.Q1(t.obs, t.action) - Q_target) ** 2)
        loss_Q2 = torch.mean((self.Q2(t.obs, t.action) - Q_target) ** 2)

        self.optimizer_Q1.zero_grad()
        loss_Q1.backward()
        self.optimizer_Q1.step()

        self.optimizer_Q2.zero_grad()
        loss_Q2.backward()
        self.optimizer_Q2.step()

        # update actor
        action_sample, logProb_sample, mu_sample = self.actor(t.obs)
        Q = torch.min(self.Q1(t.obs, action_sample),
                      self.Q2(t.obs, action_sample))
        objective_actor = torch.mean(Q - self.alpha * logProb_sample)
        loss_actor = -objective_actor

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        with torch.no_grad():
            for p, p_tar in zip(self.Q1.parameters(), self.Q1_tar.parameters()):
                p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))
            for p, p_tar in zip(self.Q2.parameters(), self.Q2_tar.parameters()):
                p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))

    def act_probabilistic(self, obs: torch.Tensor):
        self.actor.eval()
        a, logProb, mu = self.actor(obs)
        self.actor.train()
        return a

    def act_deterministic(self, obs: torch.Tensor):
        self.actor.eval()
        a, logProb, mu = self.actor(obs)
        self.actor.train()
        return mu


class ActorNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        self.ln2pi = torch.log(Tensor([2*pi_constant]))

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output_mu = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output_mu.weight)
        torch.nn.init.zeros_(self.output_mu.bias)

        self.output_logsig = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output_logsig.weight)
        torch.nn.init.zeros_(self.output_logsig.bias)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        mu = self.output_mu(x)
        sig = torch.exp(self.output_logsig(x))

        # for the log probability under tanh-squashed Gaussian, see Appendix C of the SAC paper
        u = mu + sig * torch.normal(torch.zeros(size=mu.shape), 1)
        a = torch.tanh(u)
        logProbu = -1/2 * (torch.sum(torch.log(sig**2), dim=1, keepdims=True) +
                           torch.sum((u-mu)**2/sig**2, dim=1, keepdims=True) +
                           a.shape[1]*self.ln2pi)
        logProba = logProbu - torch.sum(torch.log(1 - a ** 2 + 0.000001), dim=1, keepdims=True)
        return a, logProba, torch.tanh(mu)


class QCriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


env_directory = 'path-to-the-env/3Dball_single_no_learn_colored.app'
unity_env = UnityEnvironment(env_directory, no_graphics=True)
env = UnityToGymWrapper(unity_env, 0, allow_multiple_obs=True)

config = {
    'dim_obs': 8,
    'dim_action': env.action_space.shape[0],
    'dims_hidden_neurons': (120, 120),
    'lr': 0.01,
    'smooth': 0.99,
    'discount': 0.99,
    'alpha': 0.2,
    'batch_size': 32,
    'replay_buffer_size': 20000,
    'seed': 1,
    'max_episode': 500,
}

sac = SAC(config)
buffer = ReplayBuffer(config)


def get_avg_scores(ep_rewards):
    x = [i + 1 for i in range(len(ep_rewards))]
    avg_score = np.zeros(len(ep_rewards))
    for i in range(len(avg_score)):
       if i < 20:
            avg_score[i] = np.mean(ep_rewards[0:(i + 1)])
       else:
            avg_score[i] = np.mean(ep_rewards[i - 20:(i + 1)])
    return x, avg_score


steps = 0
plt.title('average 20 scores')
plt.ion()
plt.show()
ep_rewards = []
steps = 0
for i_episode in range(config['max_episode']):
    obs = env.reset()
    obs = obs[0]
    done = False
    t = 0
    ret = 0.
    while done is False:
        #env.render()

        obs_tensor = torch.tensor(obs).type(Tensor)
        action = sac.act_probabilistic(obs_tensor[None, :]).detach().numpy()[0, :]
        next_obs, reward, done, info = env.step(action)
        next_obs = next_obs[0]
        #obs = obs[0]
        #action = action[0]
        buffer.append_memory(obs=obs_tensor,
                             action=torch.from_numpy(action),
                             reward=torch.from_numpy(np.array([reward/10.0])),
                             next_obs=torch.from_numpy(next_obs).type(Tensor),
                             done=done)

        sac.update(buffer)

        t += 1
        steps += 1
        ret += reward

        obs = copy.deepcopy(next_obs)
        if steps % 1000 == 0:
                print("steps = ", steps)
        if done:
            print("Episode {} return {}".format(i_episode, ret))
            ep_rewards.append(ret)
            x ,avg_score = get_avg_scores(ep_rewards)
            plt.plot(x, avg_score)
            plt.pause(0.05)

env.close()
