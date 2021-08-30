import random

from SBAgent import SBAgent
from SBEnviornment.SBEnvironmentWrapper import SBEnvironmentWrapper


class RandomAgent(SBAgent):
    def __init__(self, env: SBEnvironmentWrapper, level_selection_function, id: int = 28888, level_list: list = [], ):
        SBAgent.__init__(self, level_list=level_list, env=env, id=id)
        # initialise a record of the levels to the agent

        self.id = id
        self.env = env  # used to sample random action
        self.level_selection_function = level_selection_function
        self.state_representation_type = 'symbolic'
        self.episode_rewards = {}
        self.did_win = {}

    def select_level(self):
        # you can choose to implement this by yourself, or just get it from the LevelSelectionSchema
        idx = self.level_selection_function(self.total_score_record)
        return idx

    def select_action(self, state, mode=None):
        shot = [random.randint(-200, -10), random.randint(-200, 200), random.randint(50, 80)]
        return shot

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
