import gym
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class OpenAICustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(OpenAICustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.feature_head = models.resnet18(pretrained=True)
        self.feature_head.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=2, padding=3)
        self.feature_head.fc = nn.Linear(512, features_dim)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.feature_head(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.feature_head(observations)