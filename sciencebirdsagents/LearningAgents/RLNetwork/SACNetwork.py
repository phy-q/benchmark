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


class SACNetwork:

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
        self.value = ValueNetwork(h, w).to(device)
        self.target_value = ValueNetwork(h, w).to(device)
        self.scale = reward_scale
        self.if_save_local = if_save_local

        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.value_criterion = nn.SmoothL1Loss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_1_criterion = nn.SmoothL1Loss()
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.critic_2_criterion = nn.SmoothL1Loss()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.value.eval()

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.value.train()

    def update_value_model(self, tau: float):
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def train_model_loader(self, reward_scale, total_train_time: int, train_time: int, loader: DataLoader, gamma=0.99,
                           tau=1e-2):
        self.train()
        self.scale = reward_scale
        clip_value = 0.5
        for _ in range(train_time):
            value_loss_list = []
            critic_loss_list1 = []
            critic_loss_list2 = []
            actor_loss_list = []
            for batch in loader:
                state, action, next_state, reward, is_done = batch

                reward_batch = reward.to(self.device).float().view(-1, 1) - 0.5
                action_batch = action.to(self.device).float()
                state = state.float().to(self.device)
                next_state = next_state.float().to(self.device)
                is_done = is_done.long().to(self.device).view(-1, 1)

                value = self.value(state)

                value_ = self.target_value(next_state)

                new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)

                # training Q value network
                q_hat = self.scale * reward_batch + (1 - is_done) * gamma * value_
                q1_old_policy = self.critic_1(state, action_batch)
                q2_old_policy = self.critic_2(state, action_batch)
                critic_1_loss = self.critic_1_criterion(q1_old_policy, q_hat.detach())
                critic_2_loss = self.critic_2_criterion(q2_old_policy, q_hat.detach())

                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_1.parameters(), clip_value)
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_2.parameters(), clip_value)
                self.critic_2_optimizer.step()
                critic_loss_list1.append(critic_1_loss.detach().cpu().item())
                critic_loss_list2.append(critic_2_loss.detach().cpu().item())

                # train value network
                q1_new_policy = self.critic_1(state, new_actions)
                q2_new_policy = self.critic_2(state, new_actions)
                critic_value = torch.min(q1_new_policy, q2_new_policy)
                value_target = critic_value - log_probs
                value_loss = self.value_criterion(value, value_target.detach())
                self.value_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.value.parameters(), clip_value)
                self.value_optimizer.step()
                value_loss_list.append(value_loss.detach().cpu().item())

                # train action network
                actor_loss = torch.mean(log_probs - critic_value.detach())
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), clip_value)
                self.actor_optimizer.step()
                actor_loss_list.append(actor_loss.detach().cpu().item())

            self.writer.add_scalar('training_critic1_loss', np.average(critic_loss_list1), total_train_time + _)
            self.writer.add_scalar('training_critic2_loss', np.average(critic_loss_list2), total_train_time + _)

            self.writer.add_scalar('training_actor_loss', np.average(actor_loss_list), total_train_time + _)
            self.writer.add_scalar('training_value_loss', np.average(value_loss_list), total_train_time + _)
        self.update_value_model(tau)

    def save_model(self, param_name, step):
        print('saving models')
        models_to_save = [self.actor, self.value, self.target_value, self.critic_1, self.critic_2]
        models_name = ['actor', 'value', 'target_value', 'critic_1', 'critic_2']
        for i, model in enumerate(models_to_save):
            path_name = param_name + "_" + models_name[i]
            model.save_model("LearningAgents/saved_model/{}_{}.pt".format(path_name, step))

    def load_model(self, param_name, step):
        print('loading models')
        models_to_save = [self.actor, self.value, self.target_value, self.critic_1, self.critic_2]
        models_name = ['actor', 'value', 'target_value', 'critic_1', 'critic_2']
        for i, model in enumerate(models_to_save):
            if "_" not in model.checkpoint_file:
                path_name = param_name + "_" + models_name[i]
                model.load_model("LearningAgents/saved_model/{}_{}.pt".format(path_name, step))

        # todo finish it later

    def transform_sparse(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image_sparse(h=self.h, w=self.w)

    def transform_full(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image(h=self.h, w=self.w)

    def transform(self, state):
        return self.transform_full(state) if self.if_save_local else self.transform_sparse(state)
