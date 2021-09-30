import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from LearningAgents.RLNetwork.DQNSymbolicBase import DQNSymbolicBase
from einops import rearrange


class DQNRelationalSymbolic(DQNSymbolicBase):

    def __init__(self, h, w, outputs, if_save_local=False, writer=None, device='cpu'):
        super(DQNRelationalSymbolic, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs,
                                                      if_save_local=if_save_local)

        #####
        self.input_type = 'symbolic'
        self.output_type = 'discrete'
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        #####

        self.conv1_ch = 1
        self.conv2_ch = 512  # dim `F` in paper
        self.conv3_ch = 24
        self.conv4_ch = 30
        self.H = h
        self.W = w
        self.node_size = 64  # entity embedding size
        self.lin_hid = 100
        self.out_dim = outputs  # actions
        self.outputs = outputs
        self.ch_in = 12
        self.sp_coord_dim = 2
        self.N = int(self.H * self.W)
        self.n_heads = 4

        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=2, padding=3)
        self.feature_head = torch.nn.Sequential(*(list(model.children())[:-2]))
        # Compute shape by doing one forward pass
        with torch.no_grad():
            _, out_ch, out_h, out_w = self.feature_head(torch.randn(1, 12, h, w)).shape

        self.proj_shape = (out_ch + self.sp_coord_dim, self.n_heads * self.node_size)

        # Multihead attention
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.N = out_h * out_w
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
        x = self.feature_head(x)
        with torch.no_grad():
            _, _, cH, cW = x.shape
            xcoords = torch.arange(cW).repeat(cH, 1).float().to(x.device) / cW
            ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float().to(x.device) / cH
            spatial_coords = torch.stack([xcoords, ycoords], dim=0).to(x.device)
            spatial_coords = spatial_coords.unsqueeze(dim=0).to(x.device)
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
