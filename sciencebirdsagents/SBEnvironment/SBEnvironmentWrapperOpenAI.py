import json
import os
import random
import socket
import time

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from Client.agent_client import AgentClient
from Client.agent_client import GameState
from StateReader.SymbolicStateDevReader import SymbolicStateDevReader, NotVaildStateError
from StateReader.game_object import GameObjectType
from Utils.point2D import Point2D
from Utils.trajectory_planner import SimpleTrajectoryPlanner

N_DISCRETE_ACTIONS = 180
N_CHANNELS_SYMBOLIC = 12
N_CHANNELS_IMAGE = 3
HEIGHT, WIDTH = 120, 160


class SBEnvironmentWrapperOpenAI(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, level_list, action_type, state_repr_type, **kwargs):
        super(SBEnvironmentWrapperOpenAI, self).__init__()
        with open('Utils/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.ar = AgentClient(**sc_json_config[0])

        if action_type == 'discrete':
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        elif action_type == 'continuous':
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            raise NotImplementedError('{} is not a valid action type'.format(action_type))

        if state_repr_type == 'symbolic':
            self.request_state = self.ar.get_symbolic_state_without_screenshot
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(N_CHANNELS_SYMBOLIC, HEIGHT, WIDTH), dtype=np.uint8)
        elif state_repr_type == 'image':
            self.request_state = self.ar.do_screenshot
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(N_CHANNELS_IMAGE, HEIGHT, WIDTH), dtype=np.uint8)
        else:
            raise NotImplementedError('{} is not a valid state repr type'.format(state_repr_type))

        self.level_list = level_list

        self.env_id = env_id
        self.simulation_speed = kwargs['simulation_speed'] if 'simulation_speed' in kwargs else 100
        self.game_version = kwargs['game_version'] if 'game_version' in kwargs else 'Linux043'
        self.state_repr_type = state_repr_type
        self.reward_type = kwargs['reward_type'] if 'reward_type' in kwargs else 'passing'
        self.if_batch_state = kwargs['if_batch_state'] if 'if_batch_state' in kwargs else False
        self.max_attempts_per_level = kwargs['max_attempts_per_level'] if 'max_attempts_per_level' in kwargs else 5
        self.if_init = kwargs['if_init'] if 'if_init' in kwargs else True

        self.num_birds = 0
        self.num_pigs = 0
        self.previous_num_pigs = 0
        self.total_num_birds = 0
        self.total_num_pigs = 0
        self.reset_count = 0
        self.total_num_levels = len(level_list)
        self.level_list_ind = 0
        self.env_observation = None
        self.if_first_level = True
        self.current_level = level_list[self.level_list_ind]
        self.tp = SimpleTrajectoryPlanner()

        if self.if_init:
            self.connect_agent_to_server()

    def connect_agent_to_server(self):

        try:
            self.ar.connect_to_server()
        except socket.error as e:
            # logger.critical("Error in client-server communication: " + str(e))
            print("Error in client-server communication: " + str(e))

        self.ar.configure(self.env_id)
        self.ar.set_game_simulation_speed(self.simulation_speed)

        is_in_training_mode = False  # set to true when start_level is loaded
        # First state is always training....
        game_state = self.ar.get_game_state()
        if game_state == GameState.NEWTRAININGSET or game_state == GameState.RESUMETRAINING or game_state == GameState.NEWTRIAL:
            self.ar.ready_for_new_set()

        while not is_in_training_mode:
            game_state = self.ar.get_game_state()
            if game_state == GameState.PLAYING:
                self.ar.ready_for_new_set()
                is_in_training_mode = True
            elif (game_state == GameState.NEWTRAININGSET or game_state == GameState.RESUMETRAINING or
                  game_state == GameState.NEWTESTSET) or game_state == GameState.NEWTRIAL:
                self.ar.ready_for_new_set()

            elif (game_state == GameState.LEVEL_SELECTION or
                  game_state == GameState.MAIN_MENU or
                  game_state == GameState.EPISODE_MENU):
                self.ar.get_novelty_info()
                self.ar.load_level(self.level_list[0])

            elif game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
                novelty_likelihood = 0.9
                non_novelty_likelihood = 0.1
                novel_obj_ids = {1, -2, -398879789}
                novelty_level = 0
                novelty_description = ""
                self.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                  novelty_level, novelty_description)
            elif game_state == GameState.LOADING:
                pass

            else:
                # logger.error('initialise game environment failed, skip this agent.')
                print('initialise game environment failed, skip this agent.')
                raise socket.timeout()

        # logger.debug("Environment for start_level %s is launched and ready to play..." % str(start_level))
        # print("Environment for start_level %s is launched and ready to play..." % str(self.current_level))

    def reset(self):
        # the game run self.max_attempts_per_level times before going to the next level
        if not self.if_first_level:
            self.reset_count += 1
        else:
            self.if_first_level = False

        if self.reset_count >= self.max_attempts_per_level:
            if self.level_list_ind + 1 < self.total_num_levels:
                self.level_list_ind += 1

            self.current_level = self.level_list[self.level_list_ind]
            self.reset_count = 0


        self.ar.load_level(self.current_level)

        self.ar.fully_zoom_out()
        self.num_birds = 0
        self.total_num_birds = 0
        self.total_num_pigs = 0
        while self.total_num_birds == 0 or self.total_num_pigs == 0:
            self.env_observation = self.ar.get_symbolic_state_without_screenshot()
            self.total_num_birds = self.__get_num_birds(self.env_observation)
            self.total_num_pigs = self.__get_num_pigs(self.env_observation)
            time.sleep(0.5)

        self.previous_num_pigs = self.__get_num_pigs(self.env_observation)
        self.num_birds = self.__get_num_birds(self.env_observation)
        state_repr = self.request_state(resize=[HEIGHT, WIDTH])
        if self.state_repr_type == 'image':
            observation = state_repr
        elif self.state_repr_type == 'symbolic':
            observation = SymbolicStateDevReader(state_repr).get_symbolic_image(HEIGHT, WIDTH)
        return observation

    def step(self, action):
        assert self.action_space.contains(action), 'action {} is not accepted'
        is_done = False
        did_win = False

        game_state = self.ar.get_game_state()
        if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
            novelty_likelihood = 0.9
            non_novelty_likelihood = 0.1
            novel_obj_ids = {1, -2, -398879789}
            novelty_level = 0
            novelty_description = ""
            self.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                              novelty_level, novelty_description)

        self.__get_slingshot_center()

        if isinstance(self.action_space, spaces.discrete.Discrete):
            ax_pixels, ay_pixels = self.__degToShot(action)
            release_point = Point2D(int(ax_pixels), int(ay_pixels))
            tap_time = 0

        elif isinstance(self.action_space, spaces.box.Box):
            mag = self.sling_mbr.height * 0.8
            release_point = Point2D(int(action[0] * mag), int(action[1] * mag))
            tap_time = int(abs(action[2]) * 100)

        abs_release_point = Point2D(int(self.sling_center.X + release_point.X),
                                    int(self.sling_center.Y - release_point.Y))

        bird_type = self.__get_bird_on_sling_type(self.env_observation)

        if abs(int(tap_time)) == 0:
            if bird_type == GameObjectType.REDBIRD:
                tap_time = 0  # start of trajectory
            elif bird_type == GameObjectType.YELLOWBIRD:
                tap_time = 65 + random.randint(0, 24)  # 65-90% of the way
            elif bird_type == GameObjectType.WHITEBIRD:
                tap_time = 50 + random.randint(0, 19)  # 50-70% of the way
            elif bird_type == GameObjectType.BLACKBIRD:
                tap_time = 0  # do not tap black bird
            elif bird_type == GameObjectType.BLUEBIRD:
                tap_time = 65 + random.randint(0, 19)  # 65-85% of the way
            else:
                tap_time = 60

        if not self.if_batch_state:
            self.ar.shoot(abs_release_point.X, abs_release_point.Y, 0, int(tap_time), 0)
        else:
            gt_frequency = 1
            batch_states = self.ar.shoot_and_record_ground_truth(abs_release_point.X, abs_release_point.Y, 0,
                                                                 int(tap_time), gt_frequency)
        self.num_birds -= 1
        self.env_observation = self.ar.get_symbolic_state_without_screenshot()
        self.num_pigs = self.__get_num_pigs(self.env_observation)
        # this block is used to make sure that if there's no bird or pig, the game should come to an end. #
        if self.num_birds == 0 or self.__get_num_pigs(self.env_observation) == 0:
            t_max = 10  # check for t_max times if the game is winning or losing
            i = 0
            while (self.num_birds == 0 or self.num_pigs == 0) and (game_state not in [GameState.WON, GameState.LOST]):
                game_state = self.ar.get_game_state()
                if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
                    novelty_likelihood = 0.9
                    non_novelty_likelihood = 0.1
                    novel_obj_ids = {1, -2, -398879789}
                    novelty_level = 0
                    novelty_description = ""
                    self.ar.report_novelty_likelihood(novelty_likelihood, non_novelty_likelihood, novel_obj_ids,
                                                      novelty_level, novelty_description)
                time.sleep(0.5)
                i += 1
                if i > t_max:
                    # logger.warning('level stuck and reloaded')
                    print('level stuck and reloaded')
                    return self.reset()
        #####################################################################################################

        if game_state == GameState.WON or game_state == GameState.LOST:
            is_done = True
            if game_state == GameState.WON:
                did_win = True

        if (game_state == GameState.NEWTRAININGSET or
                game_state == GameState.RESUMETRAINING or
                game_state == GameState.NEWTESTSET or
                game_state == GameState.NEWTRIAL):
            self.ar.ready_for_new_set()

        info = {'did_win': did_win, 'score': self.ar.get_current_score()}
        if self.if_batch_state:
            info['batch_states'] = batch_states

        if self.state_repr_type == 'symbolic':
            observation = SymbolicStateDevReader(self.env_observation).get_symbolic_image(HEIGHT, WIDTH)
        elif self.state_repr_type == 'image':
            observation = self.request_state(resize=[HEIGHT, WIDTH])

        if self.reward_type == 'num_pigs':
            step_reward = self.previous_num_pigs - self.num_pigs
            self.previous_num_pigs = self.num_pigs
        elif self.reward_type == 'num_pigs_normalised':
            if game_state == GameState.WON:
                step_reward = 1
            else:
                step_reward = (self.previous_num_pigs - self.num_pigs) / self.total_num_pigs - 1 / self.total_num_birds
            self.previous_num_pigs = self.num_pigs
        elif self.reward_type == 'passing':
            step_reward = int(did_win)
        else:
            raise NotImplementedError(
                'reward_type: {} is not implemented, please implement it in the Config.py'.format(self.reward_type))
        return observation, step_reward, is_done, info

    def __get_num_birds(self, state):
        birds = SymbolicStateDevReader(state).find_birds()
        count = 0
        if not birds:
            return count
        for bird_type, bird_objects in birds.items():
            count += len(bird_objects)
        return count

    def __get_num_pigs(self, state):
        pigs = SymbolicStateDevReader(state).find_pigs()
        if pigs:
            return len(pigs)
        return 0

    def __get_bird_on_sling_type(self, state):
        reader = SymbolicStateDevReader(state)
        birds = reader.find_birds()
        sling = reader.find_slingshot()[0]

        return SymbolicStateDevReader(state).find_bird_on_sling(birds, sling).type

    def __get_slingshot_center(self):
        try:
            ground_truth = self.ar.get_symbolic_state_without_screenshot()
            ground_truth_reader = SymbolicStateDevReader(ground_truth)
            sling = ground_truth_reader.find_slingshot()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.tp.get_reference_point(sling)
            self.sling_mbr = sling

        except NotVaildStateError:
            self.ar.fully_zoom_out()
            ground_truth = self.ar.get_symbolic_state_without_screenshot()
            ground_truth_reader = SymbolicStateDevReader(ground_truth)
            sling = ground_truth_reader.find_slingshot()[0]
            sling.width, sling.height = sling.height, sling.width
            self.sling_center = self.tp.get_reference_point(sling)
            self.sling_mbr = sling

    def __degToShot(self, deg):
        mag = self.sling_mbr.height * 0.8
        deg = deg + 90
        ax_pixels = mag * np.cos(np.deg2rad(deg))
        ay_pixels = mag * np.sin(np.deg2rad(deg))
        out = [ax_pixels, ay_pixels]
        return out


if __name__ == '__main__':
    from SBEnvironment.Server import Server
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from LearningAgents.RLNetwork.OpenAICustomCNN import OpenAICustomCNN
    from Utils.utils import sample_levels, make_env
    os.chdir('/home/ssd1/phd-research/sciencebirdsframework/sciencebirdsagents')
    env_id, level_ind, action_type, state_repr_type = 1, 1, 'discrete', 'symbolic'
    game_version = 'Linux043'
    if_head = 'True'
    game_server = Server(state_repr_type, game_version=game_version, if_head=if_head)

    game_server.start()
    time.sleep(1)
    #env = SBEnvironmentWrapperOpenAI(level_ind, [level_ind], action_type, state_repr_type, max_attempts_per_level=5)
    env = SubprocVecEnv([make_env(env_id=env_id,
                                  level_list=[level_ind],
                                  action_type=action_type,
                                  max_attempts_per_level=1,
                                  state_repr_type=state_repr_type) for env_id, level_ind in
                         zip(range(5),
                             range(1, 5 + 1))])
    obs = env.reset()

    # It will check your custom environment and output additional warnings if needed



    policy_kwargs = dict(
        features_extractor_class=OpenAICustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    game_server.start()
    time.sleep(2)
    env = SubprocVecEnv([make_env(env_id=env_id,
                                  level_list=[level_ind],
                                  action_type=action_type,
                                  max_attempts_per_level=1,
                                  state_repr_type=state_repr_type) for env_id, level_ind in
                         zip(range(5),
                             range(1, 5 + 1))])

    #obs = env.reset()

    model = PPO(policy='CnnPolicy', device='cuda:1', n_steps=2, batch_size=2,
                create_eval_env=True, tensorboard_log='OpenAIPPO',
                env=env, verbose=1, policy_kwargs=policy_kwargs, )

    # model = DQN(policy='CnnPolicy', device='cuda:1', batch_size=32, buffer_size=10000,
    #            learning_starts=180, target_update_interval=4, gradient_steps=32,
    #            exploration_fraction = 0.01,
    #            env=env, verbose=1, policy_kwargs=dict(normalize_images=True),)

    model.learn(32)
    game_server.close()
    time.sleep(1)
    game_server.start()
    env = SubprocVecEnv([make_env(env_id, level_ind) for env_id, level_ind in zip(range(3, 5), range(3, 5))])
    model.set_env(env)
    model.learn(32)
    game_server.close()

    # todo: 1.set env, done
    #  2. normalise input of image and set image size, done
    #  3. evaluate envs, need to be separated
    #  4. tensorboard logger, done
    #  5. customer cnn, done
    #  6. add bt to info done
