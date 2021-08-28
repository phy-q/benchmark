import json

from Client.agent_client import AgentClient
from SBEnviornment import SBEnvironmentWrapper
from Utils.trajectory_planner import SimpleTrajectoryPlanner


class SBAgent:
    def __init__(self, env: SBEnvironmentWrapper, level_list: list = [], id: int = 28888):
        # Wrapper of the communicating messages
        with open('Utils/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.ar = AgentClient(**sc_json_config[0])
        self.id = id
        self.myid = 0
        self.level_list = level_list
        self.level_selection_function = None
        self.total_score_record = dict(
            [(i, {'total_score': 0, 'did_win': False, 'attempts': 0}) for i in level_list])

        self.action_type = None
        self.tp = SimpleTrajectoryPlanner()
        self.env = env
        self.episode_rewards = {}
        self.did_win = {}
        self.state_representation_type = None


    def select_level(self):
        """

        :rtype: next start_level to play
        """
        raise NotImplementedError

    def select_action(self, state, mode=None):
        """

        :type state: given a state, determine an select_action
        """
        raise NotImplementedError

    def update_score(self, level, total_score, did_win):
        """
        Update the total_score and did_win if total_score is higher or the agent won the start_level
        """

        if not self.total_score_record[level]['did_win'] and did_win:
            self.total_score_record[level]['total_score'] = total_score
            self.total_score_record[level]['did_win'] = did_win

        elif (self.total_score_record[level]['did_win'] and did_win) or (
                not self.total_score_record[level]['did_win'] and not did_win):
            if total_score > self.total_score_record[level]['total_score']:
                self.total_score_record[level]['total_score'] = total_score

        self.total_score_record[level]['attempts'] += 1

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
