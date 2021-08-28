from LearningAgents.RLNetwork.DQNBase import DQNBase
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

from StateReader.SymbolicStateReader import SymbolicStateReader


class DQNSymbolicResNet(DQNBase):

    def __init__(self, h, w, outputs, writer=None, device='cpu'):
        super(DQNSymbolicResNet, self).__init__(h=h, w=w, outputs=outputs, device=device, writer=writer)

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
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
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

    def transform(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image_sparse(h=self.h, w=self.w)
