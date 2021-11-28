from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from LearningAgents.Memory import ReplayMemory, PrioritizedReplayMemory, PrioritizedReplayMemorySumTree

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_done'))


class DQNBase(nn.Module):
    def __init__(self, h, w, device, outputs, if_save_local, writer=None):
        super(DQNBase, self).__init__()
        self.device = device
        self.writer = writer
        self.outputs = outputs
        self.h = h
        self.w = w
        self.if_save_local = if_save_local

        self.eps = None  # importance factor
        self.input_type = None  # "image" or "symbolic" or "both"

    def transform(self, state):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def train_model_loader(self, target_net: torch.nn.Module, total_train_time: int, train_time: int,
                           gamma: float, loader: DataLoader, optimizer: torch.optim, lr=0.001, sample_eps=1):
        self.train()
        target_net.train()

        optimizer = optimizer(self.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss(reduction='none')

        # eps for priorities memory replay
        self.eps = sample_eps
        for _ in range(train_time):
            ep_loss = []
            for batch in loader:
                state, action, next_state, reward, is_done = batch

                action_batch = action.long().to(self.device).view(-1, 1)
                reward_batch = reward.to(self.device).float()  # for peb to work better
                state = state.float().to(self.device)
                next_state = next_state.float().to(self.device)
                state_action_values = self(state).gather(1, action_batch)
                next_state_values = target_net(next_state).max(1)[0].detach().view(-1, 1)

                next_state_values[is_done] = 0.0
                expected_state_action_values = (next_state_values * gamma) + reward_batch.view(-1, 1)

                loss = criterion(state_action_values, expected_state_action_values)
                processed_loss = torch.mean(loss)
                # Optimize the model
                optimizer.zero_grad()
                processed_loss.backward()
                for param in self.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                ep_loss.append(processed_loss.detach().cpu().item())

            self.writer.add_scalar('training_td_loss', np.median(ep_loss), total_train_time + _)

    def train_model_memory(self, target_net: torch.nn.Module, total_train_time: int, train_time: int, train_batch: int,
                           gamma: float, memory: ReplayMemory, optimizer: torch.optim, lr=0.001, sample_eps=1):
        self.train()
        target_net.train()

        action_batch_distribution_solved = []
        action_batch_distribution_notsolved = []
        action_batch_distribution_overall = []

        optimizer = optimizer(self.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss(reduction='none')
        scheduler = StepLR(optimizer, step_size=train_time // 10, gamma=0.9)

        # eps for priorities memory replay
        self.eps = sample_eps
        ep_loss = []
        for _ in range(train_time):
            if isinstance(memory, PrioritizedReplayMemory):
                transitions, importance, sample_indices = memory.sample(train_batch, priority_scale=self.eps)
                batch = Transition(*zip(*transitions))

            elif isinstance(memory, PrioritizedReplayMemorySumTree):
                sample_indices, transitions, importance = memory.sample(train_batch)
                batch = Transition(*zip(*transitions))

            elif isinstance(memory, ReplayMemory):
                transitions = memory.sample(train_batch)
                batch = Transition(*zip(*transitions))

            if self.input_type == 'image':
                state_batch = torch.zeros((train_batch, 3, self.h, self.w))
                next_state_batch = torch.zeros((train_batch, 3, self.h, self.w))
                for state_idx in range(train_batch):
                    # import matplotlib.pylab as plt
                    # plt.imshow(np.moveaxis(np.array(batch.state[state_idx])*127.5+127.5,0,-1).astype(int))
                    # plt.show()
                    state_batch[state_idx] = self.transform(batch.state[state_idx])
                    next_state_batch[state_idx] = self.transform(batch.next_state[state_idx])

            elif self.input_type == 'symbolic':
                state_batch = torch.zeros((train_batch, 12, self.h, self.w))
                next_state_batch = torch.zeros((train_batch, 12, self.h, self.w))
                for state_idx in range(train_batch):
                    if hasattr(batch.state[state_idx], 'toTensor'):
                        state_batch[state_idx] = batch.state[state_idx].toTensor()
                        next_state_batch[state_idx] = batch.next_state[state_idx].toTensor()
                    else:
                        state_batch[state_idx] = torch.from_numpy(batch.state[state_idx]).float()
                        next_state_batch[state_idx] = torch.from_numpy(batch.next_state[state_idx]).float()

            state_batch = state_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)

            try:
                action_batch_distribution_solved.extend(
                    np.array(batch.action, dtype=object)[np.array(batch.reward) == 1].tolist())
                action_batch_distribution_notsolved.extend(
                    np.array(batch.action, dtype=object)[np.array(batch.reward) == 0].tolist())
                action_batch_distribution_overall.extend(np.array(batch.action, dtype=object).tolist())
            except:
                pass

            action_batch = torch.Tensor(batch.action).long().to(self.device).view(-1, 1)
            reward_batch = torch.Tensor(batch.reward).to(self.device)  # for peb to work better
            is_done_batch = torch.Tensor(batch.is_done).long().to(self.device)

            state_action_values = self(state_batch).gather(1, action_batch)

            next_state_values = target_net(next_state_batch).max(1)[0].detach().view(-1, 1)
            next_state_values[is_done_batch] = 0

            expected_state_action_values = (next_state_values * gamma) + reward_batch.view(-1, 1)
            errors = expected_state_action_values - state_action_values
            # Compute Huber loss
            loss = criterion(state_action_values, expected_state_action_values)

            if isinstance(memory, PrioritizedReplayMemory):
                processed_loss = torch.mean(loss * (torch.Tensor(importance).to(self.device) ** self.eps))
            else:
                processed_loss = torch.mean(loss)

            # Optimize the model
            optimizer.zero_grad()
            processed_loss.backward()
            for param in self.parameters():
                param.grad.data.clamp_(-0.5, 0.5)
            optimizer.step()

            if isinstance(memory, PrioritizedReplayMemory) or isinstance(memory, PrioritizedReplayMemorySumTree):
                memory.set_priorities(sample_indices, errors.cpu().detach().numpy())

            scheduler.step()
            ep_loss.append(processed_loss.detach().cpu().item())
        self.writer.add_scalar('training_td_loss', np.average(ep_loss),
                               total_train_time + _)
