import os

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, h, w, n_actions=3):
        super(CriticNetwork, self).__init__()
        self.h = h
        self.w = w
        self.n_actions = n_actions
        self.input_type = 'symbolic'
        self.output_type = 'continuous'

        self.feature_head = nn.Sequential(
            nn.Conv2d(12, 1, kernel_size=1, stride=1),  # 1x1 conv to find obj type.
            nn.ReLU(),
            nn.LayerNorm((self.h, self.w)),
            nn.Conv2d(1, 512, kernel_size=(self.h, self.w), stride=1),
            nn.ReLU(),
            nn.LayerNorm((512,1,1)),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LayerNorm((256,1,1)),
            nn.Flatten(),
        )

        self.q = nn.Sequential(
            nn.Linear(256 + n_actions, 1)
        )

    def forward(self, state, action):
        feature = self.feature_head(state)
        action_value = self.q(torch.cat((feature, action), dim=1))
        return action_value

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        torch.load(model_path)
