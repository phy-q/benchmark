import random

import numpy as np

from SBAgent import SBAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.LevelSelection import LevelSelectionSchema
from StateReader.SymbolicStateDevReader import SymbolicStateDevReader
from Utils.point2D import Point2D

class CollectionAgent(SBAgent):
    def __init__(self, env: SBEnvironmentWrapper, game_level_idx,
                 template, degrees_to_shoot = None, mode='Degree', agent_id=28888):
        level_list = [game_level_idx]
        SBAgent.__init__(self, level_list=level_list, env=env, id=agent_id)
        # initialise a record of the levels to the agent
        self.game_level_idx = game_level_idx
        self.id = agent_id
        self.env = env
        self.template = template
        attempts = 0
        self.mode = mode
        if self.mode == 'Degree':
            attempts = 10
        elif self.mode == 'Degree+Random':
            attempts = 20
        elif self.mode == 'Random':
            attempts = 30
        else:
            raise ValueError('mode {} is not defined'.format(self.mode))
        self.level_selection_function = LevelSelectionSchema.RepeatPlay(attempts).select
        self.degrees_to_shoot = degrees_to_shoot if mode == 'Degree' else None
        self.episode_rewards = {}
        self.did_win = {}

        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))


    def select_level(self):
        # you can choose to implement this by yourself, or just get it from the LevelSelectionSchema
        idx = self.level_selection_function(self.total_score_record)
        return idx

    def select_action(self, state):
        if self.degrees_to_shoot:
            current_num_shots = self.total_score_record[self.game_level_idx]['attempts']
            degree_to_shoot = self.degrees_to_shoot[current_num_shots]
            shot = self.__degToShot(degree_to_shoot, state)
        else:
            degree_to_shoot = np.random.randint(10,160)
            shot = self.__degToShot(degree_to_shoot, state)

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

    def __degToShot(self, deg, state):
        reader = SymbolicStateDevReader(state, self.model, self.target_class)
        sling_rect = reader.find_slingshot()[0]
        mag = sling_rect.height * 2 if self.mode == 'Degree' else np.random.rand() * 2 * sling_rect.height
        deg = deg + 90
        ax_pixels = mag * np.cos(np.deg2rad(deg))
        ay_pixels = mag * np.sin(np.deg2rad(deg))
        out = [ax_pixels, ay_pixels]
        return out
