import torch
import torch.nn as nn
from torch.distributions.normal import Normal

epsilon = 1e-6


class ActorNetwork(nn.Module):
    def __init__(self, h, w, action_scale=50, n_actions=3):
        super(ActorNetwork, self).__init__()
        self.h = h
        self.w = w
        self.n_actions = n_actions
        self.action_scale = action_scale
        self.input_type = 'symbolic'
        self.output_type = 'continuous'
        self.reparam_noise = 1e-6

        self.feature_head = nn.Sequential(
            nn.Conv2d(12, 1, kernel_size=1, stride=1),  # 1x1 conv to find obj type.
            nn.ReLU(),
            nn.LayerNorm((self.h, self.w)),
            nn.Conv2d(1, 512, kernel_size=(self.h, self.w), stride=1),
            nn.ReLU(),
            nn.LayerNorm((512, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LayerNorm((256, 1, 1)),
            nn.Flatten(),
        )

        self.mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )
        self.log_std = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

    def forward(self, state):
        prob = self.feature_head(state)
        mu = self.mu(prob)
        log_std = self.log_std(prob)
        log_std = torch.clamp(log_std, min=-20, max=-5)
        # sigma = nn.functional.relu(sigma)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean

    # def to(self, device):
    #     self.action_scale = self.action_scale.to(device)
    #     return super(ActorNetwork, self).to(device)

    # def sample_normal(self, state, reparameterize=True):
    #     mu, sigma = self.forward(state)
    #     sigma = sigma.exp()
    #     prob = Normal(mu, sigma)
    #     if reparameterize:
    #         actions = prob.rsample()
    #     else:
    #         actions = prob.sample()
    #     action = torch.tanh(actions)
    #     log_probs = (prob.log_prob(actions) - torch.log(1 - action.pow(2) + self.reparam_noise)).sum(1, keepdim=True)
    #     return action * torch.tensor(self.max_action), log_probs
    #
    # def save_model(self, model_path):
    #     torch.save(self.state_dict(), model_path)
    #
    # def load_model(self, model_path):
    #     torch.load(model_path)
