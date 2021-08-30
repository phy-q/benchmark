import os
import torch
import torch.nn.functional as F
from torch.optim import Adam


import logging
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from LearningAgents.RLNetwork.ActorNetwork import ActorNetwork
from LearningAgents.RLNetwork.CriticNetwork import CriticNetwork
from LearningAgents.RLNetwork.ValueNetwork import ValueNetwork
from StateReader.SymbolicStateReader import SymbolicStateReader
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_done'))

logger = logging.getLogger("SAC Agent training")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class SACNetwork(object):
    def __init__(self, h, w, n_actions, device, reward_scale, if_save_local, lr=0.0003, writer=None, logger=logger):

        self.gamma = 0.99
        self.tau = 0.9
        self.alpha = 1
        self.device = device
        self.writer = writer
        self.h = h
        self.w = w
        self.if_save_local = if_save_local
        self.automatic_entropy_tuning = True
        self.if_save_local = if_save_local
        self.logger = logger
        self.input_type = 'symbolic'  # "image" or "symbolic" or "both"
        self.device = device
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        self.critic1 = CriticNetwork(h, w, n_actions).to(device)
        self.critic2 = CriticNetwork(h, w, n_actions).to(device)

        self.critic_optim1 = Adam(self.critic1.parameters(), lr=3e-4)
        self.critic_optim2 = Adam(self.critic2.parameters(), lr=3e-4)

        self.critic_target1 = CriticNetwork(h, w, n_actions).to(device)
        self.critic_target2 = CriticNetwork(h, w, n_actions).to(device)
        hard_update(self.critic_target1, self.critic1)
        hard_update(self.critic_target2, self.critic2)


        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            #self.target_entropy = -torch.prod(torch.Tensor(n_actions).to(self.device)).item()
            self.target_entropy = -3
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=3e-4)

        self.policy = ActorNetwork(h, w, n_actions=n_actions).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)

        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train_model_loader(self, reward_scale, total_train_time: int, train_time: int, loader: DataLoader, gamma=0.99,
                           tau=1e-2):
        # Sample a batch from memory
        for t_t in range(train_time):
            critic_loss_list1 = []
            critic_loss_list2 = []
            actor_loss_list = []
            alpha_loss_list = []
            alpha_list = []
            for batch in loader:
                state, action, next_state, reward, is_done = batch

                # state_batch = torch.FloatTensor(state_batch).to(self.device)
                # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                # action_batch = torch.FloatTensor(action_batch).to(self.device)
                # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
                reward_batch = reward.to(self.device).float().view(-1, 1) - 0.5
                action_batch = action.to(self.device).float()
                state_batch = state.float().to(self.device)
                next_state_batch = next_state.float().to(self.device)
                mask_batch = is_done.long().to(self.device).view(-1, 1)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                    qf1_next_target = self.critic_target1(next_state_batch, next_state_action)
                    qf2_next_target = self.critic_target2(next_state_batch, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = reward_batch + (1-mask_batch) * self.gamma * (min_qf_next_target)
                qf1 = self.critic1(state_batch, action_batch/50)  # Two Q-functions to mitigate positive bias in the policy improvement step
                qf2 = self.critic2(state_batch, action_batch/50)  # Two Q-functions to mitigate positive bias in the policy improvement step
                # qf1 = self.critic1(state_batch, action_batch  # Two Q-functions to mitigate positive bias in the policy improvement step
                # qf2 = self.critic2(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
                qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf_loss = qf1_loss + qf2_loss
                self.critic_optim1.zero_grad()
                self.critic_optim2.zero_grad()
                qf_loss.backward()
                self.critic_optim1.step()
                self.critic_optim2.step()
                critic_loss_list1.append(qf1_loss.detach().cpu().item())
                critic_loss_list2.append(qf2_loss.detach().cpu().item())

                pi, log_pi, _ = self.policy.sample(state_batch)

                qf1_pi = self.critic1(state_batch, pi/50)
                qf2_pi = self.critic2(state_batch, pi/50)
                # qf1_pi = self.critic1(state_batch, pi)
                # qf2_pi = self.critic2(state_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()
                actor_loss_list.append(policy_loss.detach().cpu().item())

                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                    self.alpha = self.log_alpha.exp()
                    alpha_tlogs = self.alpha.clone() # For TensorboardX logs
                else:
                    alpha_loss = torch.tensor(0.).to(self.device)
                    alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

                alpha_loss_list.append(alpha_loss.detach().cpu().item())
                alpha_list.append(alpha_tlogs.detach().cpu().item())

            self.writer.add_scalar('training_critic1_loss', np.average(critic_loss_list1), total_train_time + t_t)
            self.writer.add_scalar('training_critic2_loss', np.average(critic_loss_list2), total_train_time + t_t)
            self.writer.add_scalar('training_actor_loss', np.average(actor_loss_list), total_train_time + t_t)
            self.writer.add_scalar('alpha_value', np.average(alpha_list), total_train_time + t_t)
            self.writer.add_scalar('alpha_loss', np.average(alpha_loss_list), total_train_time + t_t)

        self.update_value_model(tau)


    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic_path)
        torch.save(self.critic2.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


    def update_value_model(self, tau: float):
        for target_param, param in zip(self.critic_target1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        for target_param, param in zip(self.critic_target2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def eval(self):
        self.policy.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.policy.train()
        self.critic1.train()
        self.critic2.train()

    def transform_sparse(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image_sparse(h=self.h, w=self.w)

    def transform_full(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image(h=self.h, w=self.w)

    def transform(self, state):
        return self.transform_full(state) if self.if_save_local else self.transform_sparse(state)
