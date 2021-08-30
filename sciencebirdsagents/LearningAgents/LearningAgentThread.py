import csv
import logging
import os
import socket
import threading
import time
from typing import List

import torch

from LearningAgents.LearningAgent import LearningAgent
from SBEnviornment.SBEnvironmentWrapper import SBEnvironmentWrapper

logger = logging.getLogger("Agent Thread")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class AgentThread(threading.Thread):
    def __init__(self, agent: LearningAgent, env: SBEnvironmentWrapper, lock: threading.Lock, mode='train',
                 simulation_speed=100, if_first=True, if_save_local=False, memory_saving_path=None):
        self.result = None
        threading.Thread.__init__(self)
        self.agent = agent
        self.env = env
        self.mode = mode
        self.lock = lock
        self.simulation_speed = simulation_speed
        self.if_first = if_first
        self.if_save_local = if_save_local
        self.memory_saving_path = memory_saving_path
        self.h = self.agent.h if hasattr(self.agent, 'h') else self.agent.network.h
        self.w = self.agent.w if hasattr(self.agent, 'w') else self.agent.network.w

    def save_local(self, tran_s0, action_idx, tran_s, r, is_done):
        if not os.path.exists(self.memory_saving_path):
            os.mkdir(self.memory_saving_path)

        if not os.path.exists(os.path.join(self.memory_saving_path, 'memory_meta.csv')):
            with open(os.path.join(self.memory_saving_path, 'memory_meta.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["state_path", "action", "nextstate_path", "reward", "is_done"])
            with open(os.path.join(self.memory_saving_path, 'memory_meta.csv'), 'a+') as f:
                writer = csv.writer(f)
                with open(os.path.join(self.memory_saving_path, 'memory_meta.csv'), 'r') as fr:
                    idx = len(list(csv.reader(fr)))
                state_path = os.path.join(self.memory_saving_path, "state_{}.pt".format(idx))
                try:
                    action = action_idx.item()
                except:
                    action = action_idx
                nextstate_path = os.path.join(self.memory_saving_path, "nextstate_{}.pt".format(idx))
                reward = r
                is_done = is_done
                to_write = [state_path, action, nextstate_path, reward, is_done]
                writer.writerow(to_write)

            # save state and next state
            torch.save(tran_s0, state_path)
            torch.save(tran_s, nextstate_path)
        else:
            with open(os.path.join(self.memory_saving_path, 'memory_meta.csv'), 'a+') as f:
                writer = csv.writer(f)
                with open(os.path.join(self.memory_saving_path, 'memory_meta.csv'), 'r') as fr:
                    idx = len(list(csv.reader(fr)))
                state_path = os.path.join(self.memory_saving_path, "state_{}.pt".format(idx))
                try:
                    action = action_idx.item()
                except:
                    action = action_idx

                nextstate_path = os.path.join(self.memory_saving_path, "nextstate_{}.pt".format(idx))
                reward = r
                is_done = is_done
                to_write = [state_path, action, nextstate_path, reward, is_done]
                writer.writerow(to_write)

            # save state and next state
            torch.save(tran_s0, state_path)
            torch.save(tran_s, nextstate_path)

    def run(self):
        self.agent.network.eval()
        try:
            self.env.make(agent=self.agent, start_level=self.agent.level_list[0],
                          state_representation_type=self.agent.state_representation_type,
                          if_first_server=self.if_first)
            s, r, is_done, info = self.env.reset()
            while True:
                eps_reward = 0
                while not is_done:
                    s0 = s
                    if self.agent.action_type == 'discrete':
                        action, action_idx = self.agent.select_action(s, self.mode)
                    elif self.agent.action_type == 'continuous':
                        state = torch.from_numpy(self.agent.network.transform(s)).to(
                            self.agent.network.device).unsqueeze(
                            0)
                        action = self.agent.select_action(state.float())
                    else:
                        raise ValueError("unknown action type {}".format(self.agent.action_type))

                    s, r, is_done, info = self.env.step(action)

                    if hasattr(self.agent, 'network'):
                        tran_s0 = self.agent.network.transform(s0)
                        tran_s = self.agent.network.transform(s)
                    else:
                        tran_s0 = self.agent.transform(s0)
                        tran_s = self.agent.transform(s)
                    if not (tran_s0 is not None) or not (tran_s is not None):
                        logger.error('one of the states is None')

                    if self.agent.action_type == 'discrete':
                        if isinstance(action_idx, torch.Tensor):
                            action_idx = action_idx.detach().cpu().numpy()  # we don't need to save action in gpu with gradient
                        with self.lock:
                            if self.if_save_local:
                                self.save_local(tran_s0, action_idx, tran_s, r, is_done)
                                self.agent.replay_memory.action_num += 1
                            else:
                                self.agent.replay_memory.push(tran_s0, action_idx,
                                                              tran_s, r, is_done)
                    else:
                        if isinstance(action, torch.Tensor):
                            action = action.detach().cpu().numpy().tolist()  # same as above
                        with self.lock:
                            if self.if_save_local:
                                self.save_local(tran_s0, action, tran_s, r, is_done)
                                self.agent.replay_memory.action_num += 1
                            else:
                                self.agent.replay_memory.push(tran_s0, action,
                                                              tran_s, r, is_done)

                    eps_reward += r

                did_win = info[0]
                total_score = info[1]
                self.agent.update_score(self.env.current_level, total_score, did_win)
                self.agent.update_episode_rewards(self.env.current_level, eps_reward)
                self.agent.update_winning(self.env.current_level, did_win)

                logger.debug("agent: {}, start_level: {} , result: {}".format(self.agent.id, self.env.current_level,
                                                                              self.agent.total_score_record[
                                                                                  self.env.current_level]))
                logger.debug("replay_memory length: {}".format(len(self.agent.replay_memory)))
                self.env.current_level = self.agent.select_level()
                if not self.env.current_level:  # that's when all the levels has been played.
                    logger.debug('agent {} finished running'.format(self.agent.id))
                    return
                s, r, is_done, info = self.env.reload_current_level()
        except socket.timeout:
            logger.error("sever response timeout, stop the agent")
            return


# Multithread agents manager
# can be used to collect trajectories of the same agent by using multiple instances
class MultiThreadTrajCollection:

    def __init__(self, agents: List[LearningAgent], memory_saving_path=None, simulation_speed=100):
        self.agents = agents
        self.lock = threading.Lock()
        self.simulation_speed = simulation_speed
        self.memory_saving_path = memory_saving_path

    # Connects agents to the SB games and starts training
    # at the moment agents will connect to each level 1 by 1
    # i.e. agent 1 will correspond to level 1, agent 2 to level 2, etc
    def connect_and_run_agents(self, mode='train'):
        agents_threads = []
        try:
            for i in range(1, len(self.agents) + 1):
                logger.debug('agent %s running' % str(i))
                agent = AgentThread(self.agents[i - 1], self.agents[i - 1].env, self.lock, mode=mode,
                                    memory_saving_path=self.memory_saving_path,
                                    simulation_speed=self.simulation_speed, if_first=True if i == 1 else False,
                                    if_save_local=self.agents[i - 1].if_save_local if hasattr(self.agents[i - 1],
                                                                                              'if_save_local') else
                                    self.agents[i - 1].network.if_save_local)
                agent.start()
                agents_threads.append(agent)
                if i == 1:
                    time.sleep(10)
                else:
                    time.sleep(0.5)

            logger.debug('all agents connected')
            for agent in agents_threads:
                agent.join()

            logger.debug("all agents finished training")
        except Exception as e:
            logger.warning("Error in training agents: " + str(e))

        return [agent.result for agent in agents_threads]
