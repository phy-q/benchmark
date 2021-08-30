import numpy as np
import torch
import torch.nn as nn

from LearningAgents.RLNetwork.DQNSymbolicBase import DQNSymbolicBase


class DQNSymbolicDueling(DQNSymbolicBase):

    def __init__(self, h, w, outputs, if_save_local=False, writer=None, device='cpu'):
        super(DQNSymbolicDueling, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs,
                                                 if_save_local=if_save_local)
        #####
        self.input_type = 'symbolic'
        self.output_type = 'discrete'
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        #####

        self.feature_head = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=3, stride=1), kernel_size=3, stride=1))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=3, stride=1), kernel_size=3, stride=1))

        linear_input_size = convw * convh * 64

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
