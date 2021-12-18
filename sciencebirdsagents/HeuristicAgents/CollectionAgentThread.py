import csv
import cv2
import json
import numpy as np
import os
import threading
import time
from typing import List
import hickle
from HeuristicAgents.CollectionAgent import CollectionAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from StateReader.SymbolicStateDevReader import SymbolicStateDevReader


class AgentThread(threading.Thread):
    def __init__(self, agent: CollectionAgent, env: SBEnvironmentWrapper, lock: threading.Lock, mode='train',
                 simulation_speed=100, saving_path='PhyreStyleTrainingData'):
        self.result = None
        threading.Thread.__init__(self)
        self.agent = agent
        self.env = env
        self.mode = mode
        self.lock = lock
        self.simulation_speed = simulation_speed
        self.saving_path = saving_path
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))

    def save_local(self, s0, s0_image, action, if_win, attempts, obj_movements, game_level_idx, template):
        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)

        if not os.path.exists(os.path.join(self.saving_path, template)):
            os.mkdir(os.path.join(self.saving_path, template))

        game_level_save_path = os.path.join(self.saving_path, template, str(game_level_idx))
        if not os.path.exists(game_level_save_path):
            os.mkdir(game_level_save_path)

        state_path = os.path.join(os.path.join(self.saving_path, template),
                                  "{}_{}_state.pt".format(template, game_level_idx))
        if not os.path.exists(state_path):
            with open(state_path, 'w') as f:
                json.dump(s0, f)
        image_path = os.path.join(os.path.join(self.saving_path, template),
                                  "{}_{}_image.jpg".format(template, game_level_idx))
        if not os.path.exists(image_path):
            s0_image = cv2.cvtColor(s0_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, s0_image)

        obj_movements_path = os.path.join(game_level_save_path,
                                          "{}_{}_{}_{}".format(template, game_level_idx, str(action), if_win))
        with open(obj_movements_path, 'w') as f:
            json.dump(obj_movements, f)

    def convert_bts_to_obj_movements(self, bt_gts):
        obj_dict = {}
        bt_gts = [gt[0]['features'] for gt in bt_gts]
        init_gt = bt_gts[0]
        id_convert = {}
        for i, obj in enumerate(init_gt):
            if obj['properties']['label'] not in ['Ground', 'Slingshot']:
                id_convert[obj['properties']['id']] = i

        for gt in bt_gts:
            for obj in gt:
                if obj['properties']['label'] in ['Ground', 'Slingshot', 'Trajectory']:
                    continue
                obj_id = obj['properties']['id']
                if obj_id not in id_convert:
                    print('issue')
                new_id = id_convert[obj_id]
                coordinates = obj['geometry']['coordinates']

                if new_id not in obj_dict:
                    obj_dict[new_id] = [coordinates]
                else:
                    obj_dict[new_id].append(coordinates)

        return obj_dict

    def run(self):

        self.env.make(agent=self.agent, start_level=self.agent.level_list[0],
                      state_representation_type='symbolic')
        s, r, is_done, info = self.env.reset()
        while True:
            while not is_done:
                s0 = s
                s0_image = self.agent.ar.do_screenshot()
                save_prefix = self.agent.template + "_" + str(self.env.current_level)
                state_name = save_prefix + "_state"
                action, deg = self.agent.select_action(s)
                s, _, is_done, info = self.env.step(action, batch_gt=True)
            did_win = info[0]
            batch_gts = info[2]
            total_score = info[1]
            self.agent.update_score(self.env.current_level, total_score, did_win)
            self.agent.update_episode_rewards(self.env.current_level, total_score)
            self.agent.update_winning(self.env.current_level, did_win)
            attempts = self.agent.total_score_record[self.env.current_level]['attempts']

            full_image, boxes, masks = self.process_batch_gts(batch_gts)
            save_path = f'{self.saving_path}/{self.agent.template}/{self.env.current_level}'
            os.makedirs(save_path, exist_ok=True)
            with self.lock:
                # save bounding boxes
                hickle.dump(full_image, f'{save_path}/{deg:.4f}_image.hkl', mode='w', compression='gzip')
                hickle.dump(int(did_win), f'{save_path}/{deg:.4f}_label.hkl', mode='w',
                            compression='gzip')
                hickle.dump(boxes, f'{save_path}/{deg:.4f}_boxes.hkl', mode='w', compression='gzip')
                hickle.dump(masks, f'{save_path}/{deg:.4f}_masks.hkl', mode='w', compression='gzip')

            self.env.current_level = self.agent.select_level()

            if not self.env.current_level:  # that's when all the levels has been played.
                return
            s, r, is_done, info = self.env.reload_current_level()

    def process_batch_gts(self, batch_gts):

        '''
        batch_image: n x h x w array, with each type of object occupys a number
        boxes: n x n_obj x 6 ([o_id, x1, y1, x2, y2, if_destroyed])
        masks: n x n_obj x h_mask x w_mask
        '''
        input_w = 160
        input_h = 120
        im_width = 640
        im_height = 480
        mask_size = 21

        full_image = np.zeros((len(batch_gts), input_h, input_w))
        full_objs = []
        for i, gt in enumerate(batch_gts):
            symbolic_state_reader = SymbolicStateDevReader(gt, self.model, self.target_class)
            image, obj_ids = symbolic_state_reader.get_symbolic_image_flat(input_h, input_w)
            full_image[i] = image
            full_objs.append(obj_ids)

        all_ids = set()
        for objs_t in full_objs:
            [all_ids.add(obj) for obj in objs_t.keys()]

        boxes = np.zeros((len(batch_gts), len(all_ids), 6))
        masks = np.zeros((len(batch_gts), len(all_ids), mask_size, mask_size))
        for t, objs in enumerate(full_objs):
            for id_ind, id in enumerate(all_ids):
                if id in objs:
                    top_left_x, top_left_y = objs[id].top_left
                    bottom_right_x, bottom_right_y = objs[id].bottom_right



                    top_left_x *= (input_w-1) / (im_width-1)
                    bottom_right_x *= (input_w-1) / (im_width-1)
                    top_left_y *= (input_h-1) / (im_height-1)
                    bottom_right_y *= (input_h-1) / (im_height-1)

                    top_left_x = max(top_left_x, 0)
                    bottom_right_x = min(bottom_right_x, input_w-1)
                    top_left_y = max(top_left_y, 0)
                    bottom_right_y = min(bottom_right_y, input_h-1)

                    boxes[t, id_ind] = [id_ind, top_left_x, top_left_y, bottom_right_x, bottom_right_y, 1]
                    mask_im = np.zeros((input_h,input_w))

                    for x in range(np.int(top_left_x), int(np.ceil(bottom_right_x))):
                        for y in range(np.int(top_left_y), int(np.ceil(bottom_right_y))):
                            mask_im[y,x] = 1

                    masks[t, id_ind] = cv2.resize(mask_im, (mask_size, mask_size)) >= 0.5


                else:
                    boxes[t, id_ind] = [id_ind, -1, -1, -1, -1, 0]
                    mask_im = np.zeros((input_h,input_w))
                    masks[t, id_ind] = cv2.resize(mask_im, (mask_size, mask_size)) >= 0.5

        return full_image, boxes, masks


# Multithread .agents manager
# can be used to collect trajectories of the same agent by using multiple instances
class MultiThreadTrajCollection:

    def __init__(self, agents: List[CollectionAgent], simulation_speed=100):
        self.agents = agents
        self.lock = threading.Lock()
        self.simulation_speed = simulation_speed

    # Connects agents to the SB games and starts training
    # at the moment agents will connect to each level 1 by 1
    # i.e. agent 1 will correspond to level 1, agent 2 to level 2, etc
    def connect_and_run_agents(self, saving_path='PhyreStyleTrainingData', mode='train'):
        agents_threads = []
        try:
            for i in range(1, len(self.agents) + 1):
                print('agent %s running' % str(i))
                agent = AgentThread(self.agents[i - 1], self.agents[i - 1].env, self.lock, mode=mode,
                                    simulation_speed=self.simulation_speed, saving_path=saving_path)
                agent.start()
                agents_threads.append(agent)
                time.sleep(2)

            for agent in agents_threads:
                agent.join()

            print("Agents finished training")
        except Exception as e:
            print("Error in training agents: " + str(e))

        return [agent.result for agent in agents_threads]
