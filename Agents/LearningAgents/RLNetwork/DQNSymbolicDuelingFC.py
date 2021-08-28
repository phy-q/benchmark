from LearningAgents.RLNetwork.DQNBase import DQNBase
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

from StateReader.SymbolicStateReader import SymbolicStateReader


class DQNSymbolicDuelingFC(DQNBase):

    def __init__(self, h, w, outputs, writer=None, device='cpu'):
        super(DQNSymbolicDuelingFC, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs)

        #####
        self.input_type = 'symbolic'
        self.output_type = 'discrete'
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        #####

        self.feature_head = nn.Sequential(
            nn.Linear(self.w * self.h * 12,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

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
        x = torch.flatten(x,1)
        x = self.feature_head(x)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def transform(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image_sparse(h=self.h, w=self.w)
