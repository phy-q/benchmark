import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from LearningAgents.RLNetwork.DQNSymbolicBase import DQNSymbolicBase


class DQNSymbolicResNet(DQNSymbolicBase):

    def __init__(self, h, w, outputs, if_save_local=False, writer=None, device='cpu'):
        super(DQNSymbolicResNet, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs,
                                                if_save_local=if_save_local)

        #####
        self.input_type = 'symbolic'
        self.output_type = 'discrete'
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        #####

        self.feature_head = models.resnet18(pretrained=True)
        self.feature_head.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=2, padding=3)
        self.feature_head.fc = nn.Linear(512, 256)

        self.value_stream = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SELU(),
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
