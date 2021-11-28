import random

import numpy as np

from SBAgent import SBAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from StateReader.SymbolicStateDevReader import SymbolicStateDevReader
from StateReader.game_object import GameObjectType
from Utils.point2D import Point2D
from Utils.trajectory_planner import SimpleTrajectoryPlanner


class PigShooter(SBAgent):
    def __init__(self, env: SBEnvironmentWrapper, level_selection_function, id: int = 28888, level_list: list = []):
        SBAgent.__init__(self, level_list=level_list, env=env, id=id)
        # initialise a record of the levels to the agent

        self.id = id
        self.tp = SimpleTrajectoryPlanner()
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        self.env = env  # used to sample random action
        self.level_selection_function = level_selection_function
        self.state_representation_type = 'symbolic'

    def select_level(self):
        # you can choose to implement this by yourself, or just get it from the LevelSelectionSchema
        idx = self.level_selection_function(self.total_score_record)
        return idx

    def select_action(self, state, mode=None):
        symbolic_state_reader = SymbolicStateDevReader(state, self.model, self.target_class)
        if not symbolic_state_reader.is_vaild():
            print("no pig or birds found, just shoot")
            return self.env.action_space.sample()

        sling = symbolic_state_reader.find_slingshot()[0]
        sling.width, sling.height = sling.height, sling.width

        # get all the pigs
        pigs = symbolic_state_reader.find_pigs()

        # if there is a sling, then play, otherwise skip.
        if sling:
            # If there are pigs, we pick up a pig randomly and shoot it.
            if pigs:
                release_point = None
                # random pick up a pig
                pig = pigs[random.randint(0, len(pigs) - 1)]
                temp_pt = pig.get_centre_point()

                # TODO change StateReader.cv_utils.Rectangle
                # to be more intuitive
                _tpt = Point2D(temp_pt[1], temp_pt[0])

                pts = self.tp.estimate_launch_point(sling, _tpt)

                if not pts:
                    # Add logic to deal with unreachable target
                    release_point = Point2D(-600, 560)

                elif len(pts) == 1:
                    release_point = pts[0]
                elif len(pts) == 2:
                    if random.randint(0, 1) == 0:
                        release_point = pts[1]
                    else:
                        release_point = pts[0]

                # Get the release point from the trajectory prediction module
                if release_point:
                    self.tp.get_release_angle(sling, release_point)

                    birds = symbolic_state_reader.find_birds()
                    bird_on_sling = symbolic_state_reader.find_bird_on_sling(birds, sling)
                    bird_type = bird_on_sling.type

                    if bird_type == GameObjectType.REDBIRD:
                        tap_interval = 0  # start of trajectory
                    elif bird_type == GameObjectType.YELLOWBIRD:
                        tap_interval = 65 + random.randint(0, 24)  # 65-90% of the way
                    elif bird_type == GameObjectType.WHITEBIRD:
                        tap_interval = 50 + random.randint(0, 19)  # 50-70% of the way
                    elif bird_type == GameObjectType.BLACKBIRD:
                        tap_interval = 0  # do not tap black bird
                    elif bird_type == GameObjectType.BLUEBIRD:
                        tap_interval = 65 + random.randint(0, 19)  # 65-85% of the way
                    else:
                        tap_interval = 60

                    tap_time = self.tp.get_tap_time(sling, release_point, _tpt, tap_interval)
                    shot = [release_point.X - sling.X, sling.Y - release_point.Y, tap_time]
                    return shot
                else:
                    return self.env.action_space.sample(1)

            print('didn\'t find slingshot, just shoot')
            return self.env.action_space.sample()
