import os
import random
import socket
import time

import numpy as np
import torch

from Client.agent_client import GameState
from StateReader.SymbolicStateReader import SymbolicStateReader, NotVaildStateError
from Utils.point2D import Point2D

MAX_NUMBER_OF_LEVELS = 300


class ActionSpace:
    def __init__(self, action_range):
        self.x_range_min, self.x_range_max = action_range[0]
        self.y_range_min, self.y_range_max = action_range[1]

    def sample(self):
        return [random.randint(self.x_range_min, self.x_range_max),
                random.randint(self.y_range_min, self.y_range_max)]


# MIMIC SB as if it was Gym Environment (dreams...)
class SBEnvironmentWrapper:
    def __init__(self, reward_type='score', speed=100):
        self.agent = None
        self.request_state = None  # function to request state
        self.current_level = 0
        self.total_reward_per_level = 0
        self.next_state = None
        self.shots_per_level = 0
        self.step_reward = 0
        self.previous_score = 0
        self.shots_per_level = 0  # this is to deal with the case of infinite shots
        self.did_win = False
        self.is_done = False
        self.info = [self.did_win, self.total_reward_per_level]
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        self.num_birds = 0
        self.state_representation_type = None
        self.env_state = None
        self.reward_type = reward_type
        self.simulation_speed = speed

    def make(self, agent, start_level=1, state_representation_type='symbolic', mode=None):
        """
        :int start_level: the starting start_level index :str state_representation_type: the required state representation type,
        which can be symbolic or both (symbolic and image)
        """
        self.state_representation_type = state_representation_type
        self.__start_server(mode)

        time.sleep(2)
        self.agent = agent

        try:
            self.agent.ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))
        self.current_level = start_level
        info = self.agent.ar.configure(self.agent.id)
        self.agent.ar.set_game_simulation_speed(self.simulation_speed)
        if state_representation_type == 'symbolic':
            self.request_state = self.agent.ar.get_symbolic_state_without_screenshot

        elif state_representation_type == 'image':
            self.request_state = self.agent.ar.get_symbolic_state_with_screenshot

        elif state_representation_type == 'both':
            self.request_state = self.agent.ar.get_symbolic_state_with_screenshot
        else:
            raise NotImplementedError('the type: {} is not implemented'.format('state_representation_type'))

        is_in_training_mode = False  # set to true when start_level is loaded

        try:
            # First state is always training....
            game_state = self.agent.ar.get_game_state()

            if game_state == GameState.NEWTRAININGSET or game_state == GameState.RESUMETRAINING or game_state == GameState.NEWTRIAL:
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                 allowNoveltyInfo) = self.agent.ar.ready_for_new_set()

            while not is_in_training_mode:
                game_state = self.agent.ar.get_game_state()
                if game_state == GameState.PLAYING:
                    (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                     allowNoveltyInfo) = self.agent.ar.ready_for_new_set()
                    is_in_training_mode = True
                elif (game_state == GameState.NEWTRAININGSET or game_state == GameState.RESUMETRAINING or
                      game_state == GameState.NEWTESTSET):
                    (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
                     allowNoveltyInfo) = self.agent.ar.ready_for_new_set()

                elif (game_state == GameState.LEVEL_SELECTION or
                      game_state == GameState.MAIN_MENU or
                      game_state == GameState.EPISODE_MENU):
                    self.agent.ar.get_novelty_info()
                    # self.current_level = self.agent.ar.load_next_available_level()
                    self.reload_current_level()

                elif game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
                    novelty_likelihood = 0.9
                    non_novelty_likelihood = 0.1
                    novel_obj_ids = {1, -2, -398879789}
                    novelty_level = 0
                    novelty_description = ""
                    self.agent.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                            novelty_level, novelty_description)

            print("Environment for start_level %s is launched and ready to play..." % str(start_level))

        except Exception as e:
            print("Error: ", e)
        finally:
            time.sleep(5)

    def reset(self):
        """
        it reloads current start_level and return

        :return: next_state, step_reward = 0, is_done = False, and did_win = False
        """

        self.total_reward_per_level = 0
        # self.current_state = None
        self.previous_score = 0
        self.next_state = None
        self.step_reward = 0
        self.did_win = False
        self.is_done = False
        self.shots_per_level = 0

        self.reload_current_level()

        self.next_state = self.request_state()
        if self.state_representation_type == 'image':
            self.next_state = self.next_state[0]  # to just record the image

        if self.state_representation_type == 'symbolic':
            self.env_state = self.next_state
        else:
            self.env_state = self.agent.ar.get_symbolic_state_without_screenshot()

        self.__get_slingshot_center()
        self.action_space = ActionSpace([[-200, -100], [-50, 50]])

        self.info = [self.did_win, self.total_reward_per_level]
        self.num_birds = self.__get_num_birds(self.env_state)

        return self.next_state, self.step_reward, self.is_done, self.info

    # Imitates step function from gym env
    # returns next_state, total_score, done, did_win, total_score
    def step(self, action):
        """

        :torch.Tensor.long select_action: dx, dy and [tap_time] from the slingshot where the bird will be released and tapped.
             1d length 2 tensor : [dx, dy]
             or 1d length 3 tensor: [dx, dy, tap_time]

        :return: next_state, step_reward , is_done , and info. The state after the shot has been executed and start_level
        is stable.

        # TODO: the next_state may not need to be the next state when game is stable, it can be the state right after
        # making the shot

        """
        game_state = self.agent.ar.get_game_state()

        # some times game gets stuck and the agent keeps shooting
        if self.shots_per_level > 15:
            print("Error: Got stuck, reload the start_level...")
            self.shots_per_level = 0
            return self.reset()

        # # adhoc solution for the time freeze in the begin
        if self.shots_per_level == 0:
            time.sleep(0.5)

        # the server requests a report of novelty likelyhood from time to time
        if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
            novelty_likelihood = 0.9
            non_novelty_likelihood = 0.1
            novel_obj_ids = {1, -2, -398879789}
            novelty_level = 0
            novelty_description = ""
            self.agent.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                    novelty_level, novelty_description)

        self.__get_slingshot_center()

        if isinstance(action, Point2D):
            release_point = Point2D(int(action.X),
                                    int(action.Y))

        elif isinstance(action, list):
            release_point = Point2D(int(action[0]),
                                    int(action[1]))
            tap_time = action[2] if len(action) == 3 else 0

        elif isinstance(action, torch.Tensor):
            release_point = Point2D(int(action[0]),
                                    int(action[1]))
            tap_time = float(action[2]) if len(action) == 3 else 0

        else:
            raise AssertionError("action type {} not recognized".format(type(action)))

        abs_release_point = Point2D(int(self.sling_center.X + release_point.X),
                                    int(self.sling_center.Y - release_point.Y))

        tap_time = tap_time if tap_time != 0 else random.randint(50, 80) # add random tapping if no tapping time.

        self.agent.ar.shoot(abs_release_point.X, abs_release_point.Y, 0, tap_time, 0)
        self.shots_per_level += 1
        self.num_birds -= 1
        time.sleep(1)  # ad-hoc solution for game returning state before stable/winning or losing banner shows

        self.env_state = self.agent.ar.get_symbolic_state_without_screenshot()

        total_score = self.agent.ar.get_current_score()
        self.step_reward = total_score - self.previous_score
        self.previous_score = total_score
        num_pigs = self.__get_num_pigs(self.env_state)
        # print('num_birds:{}, num_pigs:{}'.format(self.num_birds, num_pigs))
        if self.num_birds == 0 or self.__get_num_pigs(self.env_state) == 0:
            t_max = 10  # check for t_max times if the game is winning or losing
            i = 0

            while (
                    self.num_birds == 0 or num_pigs == 0) and game_state != GameState.WON and game_state != GameState.LOST:
                game_state = self.agent.ar.get_game_state()
                # print('num_birds:{}, num_pigs:{}'.format(self.num_birds, num_pigs))
                if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
                    novelty_likelihood = 0.9
                    non_novelty_likelihood = 0.1
                    novel_obj_ids = {1, -2, -398879789}
                    novelty_level = 0
                    novelty_description = ""
                    self.agent.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                            novelty_level, novelty_description)
                time.sleep(0.5)
                i += 1
                if i > t_max:
                    self.num_birds = self.__get_num_birds(self.env_state)
                    num_pigs = self.__get_num_pigs(self.env_state)
                    if i > 100:  # if it has been wait for too long, skip the level
                        print('level stuck and reloaded')
                        return self.reset()

        if game_state == GameState.WON or game_state == GameState.LOST:
            self.is_done = True
            if game_state == GameState.WON:
                self.did_win = True

        if (game_state == GameState.NEWTRAININGSET or
                game_state == GameState.RESUMETRAINING or
                game_state == GameState.NEWTESTSET or
                game_state == GameState.NEWTRIAL):
            (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
             allowNoveltyInfo) = self.agent.ar.ready_for_new_set()

        self.info = [self.did_win, self.agent.ar.get_current_score()]
        if self.state_representation_type == 'symbolic':
            self.next_state = self.env_state
        elif self.state_representation_type == 'image':
            self.next_state = self.request_state()[0]

        if self.reward_type == 'score':
            pass
        elif self.reward_type == 'passing':
            self.step_reward = int(self.did_win)
        else:
            raise NotImplementedError(
                'reward_type: {} is not implemented, please implement it in the Config.py'.format(self.reward_type))

        return self.next_state, self.step_reward, self.is_done, self.info

    def reload_current_level(self):

        self.agent.ar.load_level(self.current_level)

        #####wait until fully zoomed out#####
        game_state = self.agent.ar.get_game_state()
        if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
            novelty_likelihood = 0.9
            non_novelty_likelihood = 0.1
            novel_obj_ids = {1, -2, -398879789}
            novelty_level = 0
            novelty_description = ""
            self.agent.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                    novelty_level, novelty_description)
        self.agent.ar.fully_zoom_out()
        time.sleep(2)
        #####################################

        self.previous_score = 0
        self.next_state = None
        self.step_reward = 0
        self.did_win = False
        self.is_done = False
        self.shots_per_level = 0
        self.next_state = self.request_state()
        self.env_state = self.agent.ar.get_symbolic_state_without_screenshot()
        if self.state_representation_type == 'image':
            self.next_state = self.next_state[0]  # to just record the image

        self.num_birds = self.__get_num_birds(self.env_state)
        self.info = [self.did_win, self.total_reward_per_level]
        self.total_reward_per_level = 0
        self.__get_slingshot_center()
        self.action_space = ActionSpace([[-100, -10], [-80, 80]])

        return self.next_state, self.step_reward, self.is_done, self.info

    def load_next_level(self):
        self.current_level += 1
        if self.current_level % MAX_NUMBER_OF_LEVELS == 0 and self.current_level != 0:
            self.current_level = 1
        self.total_reward_per_level = 0
        self.previous_score = 0
        self.next_state = None
        self.step_reward = 0
        self.did_win = False
        self.is_done = False
        self.shots_per_level = 0
        self.agent.ar.load_level(self.current_level)
        self.action_space = ActionSpace([[-200, -100], [-50, 50]])

    def close(self):
        server_procs = os.popen('ps -u | grep "game_playing_interface.jar"').read().split("\n")[:-1]
        for server_proc in server_procs:
            if 'grep' not in server_proc:
                print(server_proc)
                server_proc = server_proc.split(" ")
                count = 0  # the 2nd number is the pid
                for comm in server_proc:
                    if comm != "":
                        count += 1
                        if count == 2:
                            pid = comm
                os.system('kill -1 {}'.format(pid))
                print("terminated")

    def __get_num_birds(self, state):
        birds = SymbolicStateReader(state, self.model, self.target_class).find_birds()
        count = 0
        if not birds:
            return count
        for bird_type, bird_objects in birds.items():
            count += len(bird_objects)
        return count

    def __get_num_pigs(self, state):
        pigs = SymbolicStateReader(state, self.model, self.target_class).find_pigs()
        if pigs:
            return len(pigs)
        return 0

    def __get_slingshot_center(self):
        try:
            ground_truth = self.agent.ar.get_symbolic_state_without_screenshot()
            ground_truth_reader = SymbolicStateReader(ground_truth, self.model, self.target_class)
            sling = ground_truth_reader.find_slingshot()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.agent.tp.get_reference_point(sling)
            self.sling_mbr = sling

        except NotVaildStateError:
            self.agent.ar.fully_zoom_out()
            ground_truth = self.agent.ar.get_symbolic_state_without_screenshot()
            ground_truth_reader = SymbolicStateReader(ground_truth, self.model, self.target_class)
            sling = ground_truth_reader.find_slingshot()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.agent.tp.get_reference_point(sling)
            self.sling_mbr = sling

    def __start_server(self, mode=None):
        # if server already started, do nothing
        server_procs = os.popen('ps -u | grep "game_playing_interface.jar"').read().split("\n")[:-1]
        for proc in server_procs:
            if 'grep' not in proc:
                return None
        if not mode:
            if self.state_representation_type == 'symbolic':
                os.system(
                    "gnome-terminal -- bash -c \"cd ../buildgame/Linux && java -jar ./game_playing_interface.jar --headless\"")
            else:
                os.system(
                    "gnome-terminal -- bash -c \"cd ../buildgame/Linux && java -jar ./game_playing_interface.jar\"")
        elif mode == 'headless':
            os.system(
                "gnome-terminal -- bash -c \"cd ../buildgame/Linux && java -jar ./game_playing_interface.jar --headless\"")
        else:
            os.system(
                "gnome-terminal -- bash -c \"cd ../buildgame/Linux && java -jar ./game_playing_interface.jar\"")
        print("Server started...")
