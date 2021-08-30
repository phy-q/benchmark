import torch
import torch.nn as nn
import torchvision.transforms as T

from LearningAgents.RLNetwork.DQNBase import DQNBase


class DQNImage(DQNBase):

    def __init__(self, h, w, outputs, writer=None, device='cpu'):
        super(DQNImage, self).__init__(h=h, w=w, device=device, writer=writer, outputs=outputs)

        #####
        self.input_type = 'image'
        self.output_type = 'discrete'
        self.transform = T.Compose([T.ToPILImage(), T.Resize((h, w)),
                                    T.ToTensor(), T.Normalize((.5, .5, .5), (.5, .5, .5))])
        #####

        self.feature_head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=3):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=7, stride=1), kernel_size=2, stride=1)),
            kernel_size=2, stride=1))
        convh = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=7, stride=1), kernel_size=2, stride=1)),
            kernel_size=2, stride=1))

        linear_input_size = convw * convh * 128
        self.head = nn.Linear(linear_input_size, outputs)

    def transform(self, state):
        t = T.Compose([T.ToPILImage(), T.Resize((self.h, self.w)),
                       T.ToTensor(), T.Normalize((.5, .5, .5), (.5, .5, .5))])
        return t(state)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.feature_head(x)
        x = self.head(torch.flatten(x, 1))
        return x


if __name__ == '__main__':
    model = DQNImage(h=224, w=224, outputs=91)  # 90 degree + raidus

    test_data = torch.rand((32, 3, 224, 224))
    out = model(test_data)  # expect 32, 2
    print(out)
