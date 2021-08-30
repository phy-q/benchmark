import logging
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from LearningAgents.RLNetwork.ActorNetwork import ActorNetwork
from LearningAgents.RLNetwork.CriticNetwork import CriticNetwork
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


class SACNetworkAutoTemperature:

    def __init__(self, h, w, n_actions, device, reward_scale, if_save_local, lr=0.0003, writer=None, logger=logger):
        self.device = device
        self.writer = writer
        self.h = h
        self.w = w
        self.if_save_local = if_save_local
        self.logger = logger
        self.input_type = 'symbolic'  # "image" or "symbolic" or "both"

        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))

        self.actor = ActorNetwork(h, w, n_actions=n_actions, max_action=100).to(device)
        self.critic_1 = CriticNetwork(h, w, n_actions).to(device)
        self.critic_2 = CriticNetwork(h, w, n_actions).to(device)
        self.critic_1_target = CriticNetwork(h, w, n_actions).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target = CriticNetwork(h, w, n_actions).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.scale = reward_scale
        self.if_save_local = if_save_local
        self.log_alpha = torch.tensor(np.log(1)).to(device)
        self.log_alpha.requires_grad = True

        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr * 10)
        self.log_alpha_criterion = nn.SmoothL1Loss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_1_criterion = nn.SmoothL1Loss()
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.critic_2_criterion = nn.SmoothL1Loss()
        self.min_entropy = -n_actions

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.critic_1_target.eval()
        self.critic_2_target.eval()

    def update_value_model(self, tau: float):
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def train_model_loader(self, total_train_time: int, train_time: int, loader: DataLoader, gamma=0.99,
                           reward_scale=None, tau=1e-2):
        self.train()
        clip_value = 0.5
        for _ in range(train_time):
            alpha_loss_list = []
            critic_loss_list1 = []
            critic_loss_list2 = []
            actor_loss_list = []
            alpha_list = []
            for batch in loader:
                state, action, next_state, reward, is_done = batch

                reward_batch = reward.to(self.device).float().view(-1, 1)
                action_batch = action.to(self.device).float()
                state = state.float().to(self.device)
                next_state = next_state.float().to(self.device)
                is_done = is_done.long().to(self.device).view(-1, 1)
                self.critic_1_optimizer.zero_grad()
                self.critic_2_optimizer.zero_grad()
                self.log_alpha_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()

                # training Q value network
                next_actions, log_probs = self.actor.sample_normal(next_state, reparameterize=True)
                q1_new_policy_target = self.critic_1_target(next_state, next_actions)
                q2_new_policy_target = self.critic_2_target(next_state, next_actions)
                v_hat = torch.min(q1_new_policy_target, q2_new_policy_target) - self.alpha.detach() * log_probs
                q_hat = reward_batch + (1 - is_done) * gamma * v_hat
                q1_old_policy = self.critic_1(state, action_batch)
                q2_old_policy = self.critic_2(state, action_batch)
                critic_1_loss = self.critic_1_criterion(q1_old_policy, q_hat.detach())
                critic_2_loss = self.critic_2_criterion(q2_old_policy, q_hat.detach())
                critic_1_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_1.parameters(), clip_value)
                self.critic_1_optimizer.step()
                critic_2_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_2.parameters(), clip_value)
                self.critic_2_optimizer.step()
                critic_loss_list1.append(critic_1_loss.detach().cpu().item())
                critic_loss_list2.append(critic_2_loss.detach().cpu().item())

                # train actor network
                new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
                q1_new_policy = self.critic_1(state, new_actions)
                q2_new_policy = self.critic_2(state, new_actions)
                critic_value = torch.min(q1_new_policy, q2_new_policy)
                actor_loss = torch.mean(self.alpha.detach() * log_probs - critic_value.detach())
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), clip_value)
                self.actor_optimizer.step()
                actor_loss_list.append(actor_loss.detach().cpu().item())

                # update temperature
                alpha_loss = torch.mean(self.alpha * (-log_probs - self.min_entropy).detach())
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                alpha_list.append(self.alpha.detach().cpu().item())
                alpha_loss_list.append(alpha_loss.detach().cpu().item())

            self.writer.add_scalar('training_critic1_loss', np.average(critic_loss_list1), total_train_time + _)
            self.writer.add_scalar('training_critic2_loss', np.average(critic_loss_list2), total_train_time + _)
            self.writer.add_scalar('training_actor_loss', np.average(actor_loss_list), total_train_time + _)
            self.writer.add_scalar('alpha_value', np.average(alpha_list), total_train_time + _)
        self.update_value_model(tau)

    def save_model(self, param_name, step):
        print('saving models')
        models_to_save = [self.actor, self.critic_1_target, self.critic_2_target, self.critic_1, self.critic_2]
        models_name = ['actor', 'critic_1_target', 'critic_2_target', 'critic_1', 'critic_2']
        for i, model in enumerate(models_to_save):
            path_name = param_name + "_" + models_name[i]
            model.save_model("LearningAgents/saved_model/{}_{}.pt".format(path_name, step))

    def load_model(self, param_name, step):
        print('loading models')
        models_to_save = [self.actor, self.critic_1_target, self.critic_2_target, self.critic_1, self.critic_2]
        models_name = ['actor', 'critic_1_target', 'critic_2_target', 'critic_1', 'critic_2']
        for i, model in enumerate(models_to_save):
            if "_" not in model.checkpoint_file:
                path_name = param_name + "_" + models_name[i]
                model.load_model("LearningAgents/saved_model/{}_{}.pt".format(path_name, step))

    def transform_sparse(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image_sparse(h=self.h, w=self.w)

    def transform_full(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image(h=self.h, w=self.w)

    def transform(self, state):
        return self.transform_full(state) if self.if_save_local else self.transform_sparse(state)
