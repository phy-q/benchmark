import sys
import threading
import time
from typing import List

from SBAgent import SBAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper


class AgentThread(threading.Thread):
	def __init__(self, agent: SBAgent, env: SBEnvironmentWrapper, lock: threading.Lock, mode='train',
				 simulation_speed=100):
		self.result = None
		threading.Thread.__init__(self)
		self.agent = agent
		self.env = env
		self.mode = mode
		self.lock = lock
		self.simulation_speed = simulation_speed

	def run(self):
		if self.mode == 'train':
			self.env.make(agent=self.agent, start_level=self.agent.level_list[0],
						  state_representation_type=self.agent.state_representation_type)
			s, r, is_done, info = self.env.reset()
			while True:
				while not is_done:
					action = self.agent.select_action(s)
					s, _, is_done, info = self.env.step(action)
				did_win = info[0]
				total_score = info[1]
				self.agent.update_score(self.env.current_level, total_score, did_win)
				self.agent.update_episode_rewards(self.env.current_level, total_score)
				self.agent.update_winning(self.env.current_level, did_win)
				print("self.agent: {}, level: {} , result: {}".format(self.agent.id, self.env.current_level,
																	  self.agent.total_score_record[
																		  self.env.current_level]))
				self.env.current_level = self.agent.select_level()
				if not self.env.current_level:  # that's when all the levels has been played.
					return
				s, r, is_done, info = self.env.reload_current_level()


# Multithread .agents manager
# can be used to collect trajectories of the same agent by using multiple instances
class MultiThreadTrajCollection:

	def __init__(self, agents: List[SBAgent], simulation_speed=100):
		self.agents = agents
		self.lock = threading.Lock()
		self.simulation_speed = simulation_speed

	# Connects agents to the SB games and starts training
	# at the moment agents will connect to each level 1 by 1
	# i.e. agent 1 will correspond to level 1, agent 2 to level 2, etc
	def connect_and_run_agents(self, mode='train'):
		agents_threads = []
		try:
			for i in range(1, len(self.agents) + 1):
				print('agent %s running' % str(i))
				agent = AgentThread(self.agents[i - 1], self.agents[i - 1].env, self.lock, mode=mode,
									simulation_speed=self.simulation_speed)
				agent.start()
				agents_threads.append(agent)
				time.sleep(2)

			for agent in agents_threads:
				agent.join()

			print("Agents finished training")
		except Exception as e:
			print("Error in training agents: " + str(e))

		return [agent.result for agent in agents_threads]
