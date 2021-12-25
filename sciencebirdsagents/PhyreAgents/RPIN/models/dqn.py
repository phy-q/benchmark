# Written by Haozhi Qi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign

from utils.config import _C as C
from models.layers.CIN import InterNet
from models.backbones.build import build_backbone


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define private variables
        self.ve_feat_dim = C.RPIN.VE_FEAT_DIM  # visual encoder feature dimension

        # build image encoder
        self.backbone = build_backbone(C.RPIN.BACKBONE, self.ve_feat_dim, C.INPUT.IMAGE_CHANNEL)

        predictor = [nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, kernel_size=7, stride=3), nn.ReLU(),
                     nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, kernel_size=7, stride=3), nn.Flatten()]

        self.predictor = nn.Sequential(*predictor)
        self.seq_score = nn.Sequential(
                nn.Linear(self.ve_feat_dim * 2, 1),
                nn.Sigmoid()
            )

    def forward(self, x, rois=None, num_rollouts=10, g_idx=None, x_t=None, phase='train'):
        out = self.backbone(x)
        out = self.predictor(out)
        seq_score = self.seq_score(out)
        outputs = {
            'boxes': None,
            'masks': None,
            'score': seq_score.reshape(-1),
            'if_destroyed': None,
        }
        return outputs
