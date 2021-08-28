import math
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from Agents.Utils.LevelSelection import LevelSelectionSchema

from LearningAgents.LearningAgentThread import MultiThreadTrajCollection
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.Parameters import Parameters


def sample_levels(training_level_set, num_agents, agent_idx):
    '''
    given idx, return the averaged distributed levels
    '''
    n = math.ceil(len(training_level_set) / num_agents)
    total_list = []
    for i in range(0, len(training_level_set), n):
        total_list.append(training_level_set[i:i + n])
    if agent_idx >= len(total_list):
        return None
    return total_list[agent_idx]


test_templates = {'capability_1_01_cross': ['1_01_04', '1_01_05'],
                  'capability_1_02_cross': ['1_02_05', '1_02_06'],
                  'capability_2_01_cross': ['2_01_06', '2_01_07', '2_01_08', '2_01_09'],
                  'capability_2_02_cross': ['2_02_06', '2_02_07', '2_02_08', '2_02_05'],
                  'capability_2_03_cross': ['2_03_04', '2_03_05'],
                  'capability_2_04_cross': ['2_04_02', '2_04_03'],
                  'capability_3_01_cross': ['3_01_04', '3_01_06'],
                  'capability_3_02_cross': ['3_02_03', '3_02_04'],
                  'capability_3_03_cross': ['3_03_03', '3_03_04'],
                  'capability_3_04_cross': ['3_04_03', '3_04_04'],
                  'capability_3_05_cross': ['3_05_03', '3_05_04'],
                  'capability_3_06_cross': ['3_06_03', '3_06_05'],
                  'capability_3_07_cross': ['3_07_04', '3_07_05'],
                  'capability_3_08_cross': ['3_08_02'],
                  'capability_3_09_cross': ['3_09_05', '3_09_06', '3_09_07', '3_09_08']
                  }

model_paths = os.path.join('LearningAgents', 'saved_model')

for model_path in os.listdir(model_paths):
    if 'capability' in model_path:
        print('testing {}'.format(model_path))
        template_name = model_path[:-6]
        test_to_go = test_templates[template_name]
        param = Parameters(template=test_to_go, test_template=test_to_go, if_online_learning=False)
        c = config(**param.param)

        template_name = template_name + "_" + c.network.__name__ + "_" + c.multiagent.__name__ + "_offline"
        # check if file exists
        if os.path.exists('final_run/{}'.format(template_name)):
            print('{} exists, skipping.. if you want to test it again, please delete the folder in final_run'.format(
                template_name))
            continue
        writer = SummaryWriter(log_dir='final_run/{}'.format(template_name), comment=template_name)

        network = c.network(c.h, c.w, c.output, writer, c.device).to(c.device)
        network.load_state_dict(torch.load(os.path.join(model_paths, model_path), map_location='cuda:0'))
        network.eval()

        memory = c.memory_type(c.memory_size)
        episodic_rewards = []
        winning_rates = []
        level_winning_rate = {i: 0 for i in c.test_level_list}

        agents = []
        for i in range(c.num_worker):
            level_sampled = sample_levels(c.test_level_list, c.num_worker, i)
            if not level_sampled:
                continue
            env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed)
            agent = c.multiagent(id=i + 1, dqn=network, level_list=level_sampled, replay_memory=memory, env=env,
                                 level_selection_function=LevelSelectionSchema.RepeatPlay(1).select,
                                 EPS_START=0, writer=writer)
            agents.append(agent)

        am = MultiThreadTrajCollection(agents)
        am.connect_and_run_agents()
        env.close()
        time.sleep(5)

        ## evaluate the agent's learning in testing performance ##
        episodic_reward = []
        winning_rate = []
        max_reward = []
        max_winning_rate = []

        for idx in c.test_level_list:
            for agent in agents:
                try:
                    if idx in agent.level_list:
                        episodic_reward.append(np.average(agent.episode_rewards[idx]))
                        winning_rate.append(np.average(agent.did_win[idx]))
                        max_reward.append(np.max(agent.episode_rewards[idx]))
                        max_winning_rate.append(np.max(agent.did_win[idx]))
                        if int(level_winning_rate[idx]) < max(
                                agent.did_win[idx]):
                            level_winning_rate[idx] = max(
                                agent.did_win[idx])
                except IndexError:  # agent skipped level
                    episodic_reward.append(0)
                    winning_rate.append(0)

        writer.add_scalar("average_testing_episodic_rewards", np.average(episodic_reward),
                          1)
        writer.add_scalar("average_testing_winning_rates", np.average(winning_rate),
                          1)
        writer.add_scalar("max_testing_episodic_rewards", np.average(max_reward),
                          1)
        writer.add_scalar(
            "max_testing_winning_rates - level is solved",
            np.average(max_winning_rate), 1)
        # percent of task solved
        percent_task_solved = np.average(list(level_winning_rate.values()))
        writer.add_scalar("percent of testing tasks solved",
                          percent_task_solved, 1)
        writer.flush()
        torch.cuda.empty_cache()
