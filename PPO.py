import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.final = []

    def empty_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.final[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.a_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        a_mean = self.actor(state)
        cm = torch.diag(self.a_var).to(device)
        dist = MultivariateNormal(a_mean, cm)
        action = dist.sample()
        alogprob = dist.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(alogprob)
        return action.detach()

    def evaluate(self, state, action):
        a_mean = self.actor(state)
        av = self.a_var.expand_as(a_mean)
        cm = torch.diag_embed(av).to(device)
        dist = MultivariateNormal(a_mean, cm)
        alogprob = dist.log_prob(action)
        dist_e = dist.entropy()
        state_value = self.critic(state)
        return alogprob, torch.squeeze(state_value), dist_e

class PPO:
    def __init__(self, s_dim, a_dim, action_std, lr, betas, gamma, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(s_dim, a_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.pol_old = ActorCritic(s_dim, a_dim, action_std).to(device)
        self.pol_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.pol_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        rewards = []
        dis_reward = 0
        for reward, final in zip(reversed(memory.rewards), reversed(memory.final)):
            if final:
                dis_reward = 0
            dis_reward = reward + (self.gamma * dis_reward)
            rewards.insert(0, dis_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_s = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_a = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        for _ in range(80):
            # find old actions and values :
            logprobs, sv, dist_en = self.policy.evaluate(old_s, old_a)

            # this variable fines the ratio (pi theta / pi theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            advantages = rewards - sv.detach()
            s1 = ratios * advantages
            s2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(s1, s2) + 0.5*self.MseLoss(sv, rewards) - 0.01*dist_en
            # gradient
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new policy into old policy:
        self.pol_old.load_state_dict(self.policy.state_dict())

def main():
    render = False
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)          # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    # creating environment
    env_directory = 'path-to-the-env/3Dball_single_no_learn_colored.app'
    unity_env = UnityEnvironment(env_directory)
    env = UnityToGymWrapper(unity_env, 0, allow_multiple_obs=True)
    memory = ReplayBuffer()
    ppo = PPO(env.observation_space[0].shape[0], env.action_space.shape[0], action_std, lr, betas, gamma, eps_clip)
    step_ = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        state = state[0]
        score = 0
        for t in range(max_timesteps):
            step_ +=1
            # Running policy old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            state = state[0]
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.final.append(done)
            # update
            if step_ % update_timestep == 0:
                ppo.update(memory)
                memory.empty_buffer()
                step_ = 0
            score += reward
            if render:
                env.render()
            if done:
                break

        print('episode ', i_episode, 'score %.1f' % score)
if __name__ == '__main__':
    main()
