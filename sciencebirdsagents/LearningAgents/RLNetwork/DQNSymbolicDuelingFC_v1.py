import numpy as np
import torch
import torch.nn as nn

from LearningAgents.RLNetwork.DQNSymbolicBase import DQNSymbolicBase


class DQNSymbolicDuelingFC_v1(DQNSymbolicBase):

    def __init__(self, h, w, outputs, if_save_local=False, writer=None, device='cpu'):
        super(DQNSymbolicDuelingFC_v1, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs,
                                                      if_save_local=if_save_local)
        #####
        self.input_type = 'symbolic'
        self.output_type = 'discrete'
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        #####

        self.feature_head = nn.Sequential(
            nn.Conv2d(12, 1, kernel_size=1, stride=1),  # 1x1 conv to find obj type.
            nn.LeakyReLU(),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(self.h * self.w, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
        )

        linear_input_size = 256

        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, outputs)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.feature_head(x)
        x = torch.flatten(x, 1)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean())

        return qvals
