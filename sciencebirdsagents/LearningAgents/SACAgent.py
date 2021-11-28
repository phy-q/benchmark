import os

import torch
from torch.utils.tensorboard import SummaryWriter

from LearningAgents.LearningAgent import LearningAgent
from LearningAgents.Memory import ReplayMemory
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.LevelSelection import LevelSelectionSchema


class SACAgent(LearningAgent):

    def __init__(self, network, level_list: list,
                 replay_memory: ReplayMemory,
                 env: SBEnvironmentWrapper,
                 writer: SummaryWriter = None,
                 level_selection_function=LevelSelectionSchema.RepeatPlay(5).select,
                 id: int = 28888):
        LearningAgent.__init__(self, level_list=level_list, env=env, id=id, replay_memory=replay_memory, writer=writer)

        self.level_list = level_list
        self.env = env
        self.writer = writer
        self.level_selection_function = level_selection_function
        self.id = id
        self.replay_memory = replay_memory
        self.action_type = 'continuous'
        self.state_representation_type = 'symbolic'
        self.network = network

    def select_action(self, state):
        #actions, _ = self.network.actor.sample_normal(state, reparameterize=False)
        actions, _, _ = self.network.policy.sample(state)

        return actions.view(-1)


if __name__ == '__main__':
    os.chdir('../../')
    h, w, n_actions, level_list, env, writer, reward_scale, device, replay_memory = 120, 160, 3, [1, 2,
                                                                                                  3], 1, 1, 1, 'cuda:0', 1
    agent = SACAgent(h, w, n_actions, level_list, env, writer, reward_scale, device, replay_memory)
    input_data = torch.rand((32, 12, 120, 160))
    print(agent.select_action(input_data))
    print(agent.select_action(input_data).size())
