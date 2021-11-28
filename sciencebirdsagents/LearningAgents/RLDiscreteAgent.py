import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import random

from LearningAgents.LearningAgent import LearningAgent
from Utils.LevelSelection import LevelSelectionSchema
from LearningAgents.Memory import ReplayMemory
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from torch.utils.tensorboard import SummaryWriter
from LearningAgents.RLNetwork.MultiHeadRelationalModule import MultiHeadRelationalModuleImage
from einops import rearrange, reduce

class RelAgent(LearningAgent):

    def __init__(self, network, level_list: list,
                 replay_memory: ReplayMemory,
                 env: SBEnvironmentWrapper,
                 writer: SummaryWriter = None,
                 EPS_START=0.9,
                 level_selection_function=LevelSelectionSchema.RepeatPlay(5).select,
                 id: int = 28888, ):
        LearningAgent.__init__(self, level_list=level_list, env=env, id=id, replay_memory=replay_memory, writer=writer)
        self.network = network.eval()  # the network policy
        self.replay_memory = replay_memory  # replay memory obj
        self.level_selection_function = level_selection_function
        self.state_representation_type = self.network.input_type
        self.action_type = self.network.output_type
        self.EPS_START = EPS_START
        self.input_type = self.network.input_type
        self.action_selection_mode = 'argmax'  # 'sample'

    def select_action(self, state, mode='train'):
        if mode == 'train':
            sample = random.random()
            if sample > self.EPS_START:
                with torch.no_grad():
                    if self.state_representation_type == 'image':
                        state = self.network.transform(state).unsqueeze(0).to(self.network.device)
                    elif self.state_representation_type == 'symbolic':
                        state = self.network.transform(state)
                        if hasattr(state, 'toTensor'):
                            state = state.toTensor().unsqueeze(0).to(self.network.device)
                        else:
                            state = torch.from_numpy(state).float().unsqueeze(0).to(
                                self.network.device)
                    q_values = self.network(state)

                    if self.action_selection_mode == 'sample':
                        angle = torch.Tensor(np.random.choice(range(0, q_values.size(1)), 1, p=torch.nn.Softmax(1)(
                            q_values).detach().cpu().numpy().flatten()))
                    else:
                        angle = torch.argmax(q_values, 1).to('cpu')

                    out = self.__degToShot(angle).to('cpu')

                    return out, angle
            else:
                q_values = torch.rand((1, self.network.outputs))
                angle = torch.argmax(q_values, 1).to('cpu')
                out = self.__degToShot(angle).to('cpu')
                return out, angle
        else:
            with torch.no_grad():
                if self.state_representation_type == 'image':
                    state = self.network.transform(state).unsqueeze(0).to(self.network.device)
                elif self.state_representation_type == 'symbolic':
                    state = self.network.transform(state)
                    if hasattr(state, 'toTensor'):
                        state = state.toTensor().unsqueeze(0).to(self.network.device)
                    else:
                        state = torch.from_numpy(state).float().unsqueeze(0).to(
                            self.network.device)
                q_values = self.network(state)
                if self.action_selection_mode == 'sample':
                    angle = torch.Tensor(np.random.choice(range(0, q_values.size(1)), 1, p=torch.nn.Softmax(1)(
                        q_values).detach().cpu().numpy().flatten()))
                else:
                    angle = torch.argmax(q_values, 1).to('cpu')
                out = self.__degToShot(angle)
                return out, angle

    def __degToShot(self, deg):
        # deg = torch.argmax(q_values, 1) + 90
        deg = deg + 90 if self.network.outputs == 180 else 180
        ax_pixels = 200 * torch.cos(torch.deg2rad(deg.float())).view(-1, 1)
        ay_pixels = 200 * torch.sin(torch.deg2rad(deg.float())).view(-1, 1)
        out = torch.cat((ax_pixels, ay_pixels), 1)
        if out.size(0) == 1:
            return out[0]
        return out


if __name__ == '__main__':
    os.chdir('../')
    network = MultiHeadRelationalModuleImage(h=60, w=80, outputs=180)
    level_list = [1, 2, 3]
    replay_memory = ReplayMemory(1000)
    env = SBEnvironmentWrapper(reward_type='passing', speed=1, game_version='Linux', if_head=True)
    agent = RelAgent(id=1, network=network, level_list=level_list, replay_memory=replay_memory, env=env,
                                level_selection_function=LevelSelectionSchema.RepeatPlay(10).select, EPS_START=0.95)
    env.make(agent=agent, start_level=agent.level_list[0],
                          state_representation_type=agent.state_representation_type,
                          if_first_server=True)
    s, r, is_done, info = env.reset()
    # import matplotlib.pylab as plt
    # plt.imshow(s)
    # plt.show()
    # state = torch.rand((3, 640, 640))
    action = agent.select_action(s)[0]
    env.step(action)
    print(action)
    env.close()