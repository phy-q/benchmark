from LearningAgents.DQNDiscreteAgent import DQNDiscreteAgent
from LearningAgents.RLNetwork.DQNImageDueling import DQNImageDueling
from LearningAgents.RLNetwork.DQNImageResnet import DQNImageResNet
from LearningAgents.RLNetwork.DQNSymbolicDueling import DQNSymbolicDueling
from LearningAgents.RLNetwork.DQNSymbolicResnet import DQNSymbolicResNet
from LearningAgents.RLNetwork.DQNSymbolicDuelingFC import DQNSymbolicDuelingFC
from LearningAgents.RLNetwork.DQNSymbolicDuelingFC_v1 import DQNSymbolicDuelingFC_v1
from LearningAgents.RLNetwork.DQNSymbolicDuelingFC_v2 import DQNSymbolicDuelingFC_v2
import torch
from LearningAgents.RLNetwork.DQNImage import DQNImage
from LearningAgents.Memory import ReplayMemory, PrioritizedReplayMemory, PrioritizedReplayMemoryBalanced, PrioritizedReplayMemorySumTree

class Parameters:
    def __init__(self, template, if_online_learning, test_template=None):

        self.param = {
            # operating system
            'os': "Linux",
            # pytorch parameters
            'device': "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",

            # image network parameters
            'h': 120,
            'w': 160,
            'output': 180,

            # multiagent trainning parameters
            'num_update_steps': 12 if not test_template else 18,
            'num_level_per_agent': 10,
            'num_worker': 10,
            'multiagent': DQNDiscreteAgent,
            'training_attempts_per_level': 5,
            'memory_size': 100000,
            'memory_type': ReplayMemory,
            # single agent training parameters
            'singleagent': None,

            # general trainning parameters

            'train_time_per_ep': 1024,
            'train_time_rise': 1.05,
            'train_batch': 32,
            'gamma': 0.99,
            'eps_start': 0.99,
            'eps_test': 0,
            'lr': 0.0003,
            'network': DQNSymbolicResNet,
            'reward_type': 'passing',
            'simulation_speed': 100,
            'eval_freq': 5,
            'test_steps': 10,
            'online_training': if_online_learning,

            'train_template': template,  # ['1_1_1', '1_1_2'] level 1 capability 1 template 1 and 2 for training
            'test_template': template if not test_template else test_template,  # ['1_1_3', '1_1_4'] level 1 capability 1 template 3 and 4 for testing
        }
