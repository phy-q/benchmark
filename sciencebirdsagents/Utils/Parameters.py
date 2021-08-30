# from LearningAgents.RLNetwork.SACNetwork import SACNetwork
import torch
from LearningAgents.RLNetwork.DQNSymbolicDuelingFC_v2 import DQNSymbolicDuelingFC_v2
from LearningAgents.RLNetwork.DQNImageResnet import DQNImageResNet

from LearningAgents.DQNDiscreteAgent import DQNDiscreteAgent
from LearningAgents.Memory import PrioritizedReplayMemory
from LearningAgents.RLNetwork.SACNetwork_New import SACNetwork

capability_templates_dict = {
    '1_01': ['1_01_01', '1_01_02', '1_01_03'],
    '1_02': ['1_02_01', '1_02_03', '1_02_04', '1_02_05', '1_02_06'],
    '2_01': ['2_01_01', '2_01_02', '2_01_03', '2_01_04', '2_01_05', '2_01_06', '2_01_07', '2_01_08', '2_01_09'],
    '2_02': ['2_02_01', '2_02_02', '2_02_03', '2_02_04', '2_02_05', '2_02_06', '2_02_07', '2_02_08'],
    '2_03': ['2_03_01', '2_03_02', '2_03_03', '2_03_04', '2_03_05'],
    '2_04': ['2_04_04', '2_04_05', '2_04_06', '2_04_02', '2_04_03'],
    '3_01': ['3_01_01', '3_01_02', '3_01_03', '3_01_04', '3_01_06'],
    '3_02': ['3_02_01', '3_02_02', '3_02_03', '3_02_04'],
    '3_03': ['3_03_01', '3_03_02', '3_03_03', '3_03_04'],
    '3_04': ['3_04_01', '3_04_02', '3_04_03', '3_04_04'],
    '3_05': ['3_05_03', '3_05_04', '3_05_05'],
    '3_06': ['3_06_01', '3_06_04', '3_06_06', '3_06_03', '3_06_05'],
    '3_07': ['3_07_01', '3_07_02', '3_07_03', '3_07_04', '3_07_05'],
    '3_08': ['3_08_01', '3_08_02'],
    '3_09': ['3_09_01', '3_09_02', '3_09_03', '3_09_04', '3_09_07', '3_09_08']}


class Parameters:
    def __init__(self, template, level_path, game_version, test_template=None):
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
            'num_update_steps': 20 if not test_template else 40,
            'num_level_per_agent': 10,
            # todo: add an assertion to this.
            'num_worker': 20,  # make usre it is divisible by the total number of levels
            'agent': 'ppo', #DQNDiscreteAgent, #DQNDiscreteAgent, #'a2c',
            'training_attempts_per_level': 20,
            'memory_size': 100000,
            'memory_type': PrioritizedReplayMemory,

            # general trainning parameters
            'resume': False,
            'action_type': 'discrete' , #'continuous'
            'state_repr_type': 'symbolic',

            'train_time_per_ep': 10,
            'train_time_rise': 1,
            'train_batch': 32,
            'gamma': 0.99,
            'eps_start': 0.95,
            'eps_test': 0.05,
            'lr': 0.0003,
            'network': DQNSymbolicDuelingFC_v2, #DQNSymbolicDuelingFC_v2
            'reward_type': 'passing',
            'simulation_speed': 100,
            'eval_freq': 10,
            'test_steps': 1,
            'level_path': level_path,
            'game_version': game_version,

            'train_template': template,  # ['1_1_1', '1_1_2'] level 1 capability 1 template 1 and 2 for training
            'test_template': template if not test_template else test_template,
            # ['1_1_3', '1_1_4'] level 1 capability 1 template 3 and 4 for testing
        }
