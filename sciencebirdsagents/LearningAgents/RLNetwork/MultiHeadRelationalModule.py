from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import random

from LearningAgents.LearningAgent import LearningAgent
from Utils.LevelSelection import LevelSelectionSchema
from LearningAgents.Memory import ReplayMemory
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange, reduce


class MultiHeadRelationalModuleImage(nn.Module):
    def __init__(self, h, w, outputs, device, ch_in):
        super(MultiHeadRelationalModuleImage, self).__init__()
        self.device = device
        self.input_type = 'image'
        self.output_type = 'discrete'
        self.transform = T.Compose([T.ToPILImage(), T.Resize((h, w)),
                                    T.ToTensor(), T.Normalize((.5, .5, .5), (.5, .5, .5))])

        self.conv1_ch = 8
        self.conv2_ch = 10  # dim `F` in paper
        self.conv3_ch = 24
        self.conv4_ch = 30
        self.H = h
        self.W = w
        self.node_size = 64  # entity embedding size
        self.lin_hid = 100
        self.out_dim = outputs  # actions
        self.outputs = outputs
        self.ch_in = ch_in
        self.sp_coord_dim = 2
        self.N = int(self.H * self.W)
        self.n_heads = 1

        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=(7, 7), padding=1)
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=(3, 3), padding=1)
        self.feature_head = nn.Sequential(self.conv1, self.conv2)
        self.proj_shape = (self.conv2_ch + self.sp_coord_dim, self.n_heads * self.node_size)

        # Multihead attention
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            self.N = int(self.feature_head(torch.rand(size=(1, self.ch_in, self.H, self.W))).flatten().size(0)/self.conv2_ch)

        self.k_lin = nn.Linear(self.node_size, self.N)
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        with torch.no_grad():
            self.conv_map = x.clone()
        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW).repeat(cH, 1).float() / cW
        ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float() / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N, 1, 1, 1).to(x.device)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)  # batch_size, H, W, C
        x = x.flatten(1, 2)  # batch_size, HxW, C

        # key, query, value separation
        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K)

        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q)

        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V)

        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K))
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3)
        with torch.no_grad():
            self.att_map = A.cpu().clone()
        E = torch.einsum('bhfc,bhcd->bhfd', A, V)

        # collapse head dimension
        E = rearrange(E, 'b head n d -> b n (head d)')
        # B N D' . D' D -> B N D
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        # B N D -> B D
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        return y

    def train_model_memory(self, target_net: torch.nn.Module, total_train_time: int, train_time: int, train_batch: int,
                           gamma: float, memory: ReplayMemory, optimizer: torch.optim, lr=0.001, sample_eps=1):
        pass

    def transform(self, state):
        t = T.Compose([T.ToPILImage(), T.Resize((self.H, self.W)),
                       T.ToTensor(), T.Normalize((.5, .5, .5), (.5, .5, .5))])
        return t(state)

