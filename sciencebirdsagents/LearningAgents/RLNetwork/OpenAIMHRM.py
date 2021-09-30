import gym
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from LearningAgents.RLNetwork.MultiHeadRelationalModule import MultiHeadRelationalModuleImage


class OpenAIMHRM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, device = 'cuda:0', ch_in = 12):
        super(OpenAIMHRM, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels, h, w = observation_space.shape
        self.feature_head = MultiHeadRelationalModuleImage(h, w, features_dim, device, ch_in)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.feature_head(observations)
