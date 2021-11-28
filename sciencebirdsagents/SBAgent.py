import json

from Client.agent_client import AgentClient
from SBEnvironment import SBEnvironmentWrapper
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
        self.level_selection_function = None # TODO: level selection function can be an input of the init function
        self.episode_rewards = {}
        self.did_win = {}


        self.total_score_record = dict(
            [(i, {'total_score': 0, 'did_win': False, 'attempts': 0}) for i in level_list])

        self.action_type = None
        self.tp = SimpleTrajectoryPlanner()
        self.env = env

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

    def close(self):
        server_procs = os.popen('ps -u | grep "game_playing_interface.jar"').read().split("\n")[:-1]
        for server_proc in server_procs:
            if 'grep' not in server_proc:
                logger.debug(server_proc)
                server_proc = server_proc.split(" ")
                count = 0  # the 2nd number is the pid
                for comm in server_proc:
                    if comm != "":
                        count += 1
                        if count == 2:
                            pid = comm
                os.system('kill -1 {}'.format(pid))
                logger.debug("server terminated")

    def __get_num_birds(self, state):
        birds = SymbolicStateDevReader(state, self.model, self.target_class).find_birds()
        count = 0
        if not birds:
            return count
        for bird_type, bird_objects in birds.items():
            count += len(bird_objects)
        return count

    def __get_num_pigs(self, state):
        pigs = SymbolicStateDevReader(state, self.model, self.target_class).find_pigs()
        if pigs:
            return len(pigs)
        return 0

    def __get_bird_on_sling_type(self, state):
        reader = SymbolicStateDevReader(state, self.model, self.target_class)
        birds = reader.find_birds()
        sling = reader.find_slingshot()[0]

        return SymbolicStateDevReader(state, self.model, self.target_class).find_bird_on_sling(birds, sling).type

    def __get_slingshot_center(self):
        try:
            ground_truth = self.agent.ar.get_symbolic_state_without_screenshot()
            ground_truth_reader = SymbolicStateDevReader(ground_truth, self.model, self.target_class)
            sling = ground_truth_reader.find_slingshot()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.agent.tp.get_reference_point(sling)
            self.sling_mbr = sling

        except NotVaildStateError:
            self.agent.ar.fully_zoom_out()
            ground_truth = self.agent.ar.get_symbolic_state_without_screenshot()
            ground_truth_reader = SymbolicStateDevReader(ground_truth, self.model, self.target_class)
            sling = ground_truth_reader.find_slingshot()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.agent.tp.get_reference_point(sling)
            self.sling_mbr = sling