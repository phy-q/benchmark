from LearningAgents.Memory import ReplayMemory
from SBAgent import SBAgent
from SBEnviornment import SBEnvironmentWrapper
from torch.utils.tensorboard import SummaryWriter


class LearningAgent(SBAgent):
    def __init__(self, level_list: list, replay_memory: ReplayMemory, env: SBEnvironmentWrapper,
                 writer: SummaryWriter = None,
                 id: int = 28888):
        super(LearningAgent, self).__init__(level_list=level_list, env=env, id=id)
        self.replay_memory = replay_memory
        self.writer = writer
        self.policy_net = None

