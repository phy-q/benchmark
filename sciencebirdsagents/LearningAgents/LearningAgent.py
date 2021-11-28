from LearningAgents.Memory import ReplayMemory
from SBAgent import SBAgent
from SBEnvironment import SBEnvironmentWrapper
from torch.utils.tensorboard import SummaryWriter


class LearningAgent(SBAgent):
    input_type = 'symbolic'

    def __init__(self, level_list: list, replay_memory: ReplayMemory, env: SBEnvironmentWrapper,
                 writer: SummaryWriter = None, id: int = 28888):
        super(LearningAgent, self).__init__(level_list=level_list, env=env, id=id)
        self.replay_memory = replay_memory
        self.state_representation_type = None
        self.episode_rewards = {}
        self.did_win = {}
        self.writer = writer
        self.network = None

    def select_level(self):
        idx = self.level_selection_function(self.total_score_record)
        return idx

    def update_episode_rewards(self, current_level, eps_reward):
        if current_level not in self.episode_rewards:
            self.episode_rewards[current_level] = [eps_reward]
        else:
            self.episode_rewards[current_level].append(eps_reward)

    def update_winning(self, current_level, did_win):
        if current_level not in self.did_win:
            self.did_win[current_level] = [did_win]
        else:
            self.did_win[current_level].append(did_win)
