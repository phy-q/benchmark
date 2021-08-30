import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from LearningAgents.RLNetwork.DQNBase import DQNBase


class DQNImageResNet(DQNBase):

    def __init__(self, h, w, outputs, if_save_local,writer=None, device='cpu'):
        super(DQNImageResNet, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs,if_save_local=if_save_local)

        #####
        self.input_type = 'image'
        self.output_type = 'discrete'

        #####
        linear_input_size = 128

        self.feature_head = models.resnet18(pretrained=False)
        self.feature_head.fc = nn.Linear(512, linear_input_size)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, outputs)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def transform(self, state):
        t = T.Compose([T.ToPILImage(), T.CenterCrop((360, 480)), T.Resize((self.h, self.w)),
                       T.ToTensor(), T.Normalize((.5, .5, .5), (.5, .5, .5))])
        return t(state.transpose(2,1,0))

    def forward(self, x):
        x = self.feature_head(x)
        x = torch.flatten(x, 1)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean())

        return qvals


if __name__ == '__main__':
    model = DQNImageDueling(h=224, w=224, outputs=91)  # 90 degree + raidus

    test_data = torch.rand((32, 3, 224, 224))
    out = model(test_data)  # expect 32, 2
    print(out)
