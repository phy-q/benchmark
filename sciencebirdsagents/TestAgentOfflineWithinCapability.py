import math
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from LearningAgents.LearningAgentThread import MultiThreadTrajCollection
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.LevelSelection import LevelSelectionSchema
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

benchmark_capability_templates_dict = {
    '1_01': {'train': ['1_01_04', '1_01_05', '1_01_06'], 'test': ['1_01_01', '1_01_02']},
    '1_02': {'train': ['1_02_04', '1_02_05', '1_02_06'], 'test': ['1_02_01', '1_02_02']},
    '2_01': {'train': ['2_01_03', '2_01_08', '1_01_09'], 'test': ['2_01_02', '2_01_05', '2_01_06']},
    '2_02': {'train': ['2_02_03', '2_02_06', '2_02_08'], 'test': ['2_02_01', '2_02_02']},
    '2_03': {'train': ['2_03_01', '2_03_02', '2_03_05'], 'test': ['2_03_03', '2_03_04']},
    '2_04': {'train': ['2_04_02', '2_04_05', '2_04_06'], 'test': ['2_04_01', '2_04_03', '2_04_04']},
    '3_01': {'train': ['3_01_01', '3_01_02', '3_01_03'], 'test': ['3_01_04', '3_01_06']},
    '3_02': {'train': ['3_02_02', '3_02_04'], 'test': ['3_02_01', '3_02_03']},
    '3_03': {'train': ['3_03_02', '3_03_04'], 'test': ['3_03_01', '3_03_03']},
    '3_04': {'train': ['3_04_02', '3_04_04'], 'test': ['3_04_01', '3_04_03']},
    '3_05': {'train': ['3_05_02', '3_05_03', '3_05_04'], 'test': ['3_05_01', '3_05_05']},
    '3_06': {'train': ['3_06_04', '3_06_05', '3_06_06'], 'test': ['3_06_01', '3_06_03', '3_06_07']},
    '3_07': {'train': ['3_07_01', '3_07_03', '3_07_04'], 'test': ['3_07_02', '3_07_05']},
    '3_08': {'train': ['3_08_01'], 'test': ['3_08_02']},
    '3_09': {'train': ['3_09_02', '3_09_03', '3_09_08', '3_09_06'],
             'test': ['3_09_01', '3_09_04', '3_09_07', '3_09_05']},
}
capability_templates_dict = {
    '1_01': ['1_01_01', '1_01_02', '1_01_03'],
    '1_02': ['1_02_01', '1_02_03', '1_02_04', '1_02_05', '1_02_06'],
    '2_01': ['2_01_01', '2_01_02', '2_01_03', '2_01_04', '2_01_05', '2_01_06', '2_01_07', '2_01_08', '2_01_09'],
    '2_02': ['2_02_01', '2_02_02', '2_02_03', '2_02_04', '2_02_05', '2_02_06', '2_02_07', '2_02_08'],
    '2_03': ['2_03_01', '2_03_02', '2_03_03', '2_03_04', '2_03_05'],
    '2_04': ['2_04_04', '2_04_05', '2_04_06', '2_04_02', '2_04_03'],
    '3_01': ['3_01_01', '3_01_02', '3_01_03', '3_01_04', '3_01_06'],
    '3_02': ['3_02_01', '3_02_02', '3_02_03', '3_02_04'],
    '3_03': ['3_03_01', '3_03_02', '3_03_03', '3_03_04'],
    '3_04': ['3_04_01', '3_04_02', '3_04_03', '3_04_04'],
    '3_05': ['3_05_03', '3_05_04', '3_05_05'],
    '3_06': ['3_06_01', '3_06_04', '3_06_06', '3_06_03', '3_06_05'],
    '3_07': ['3_07_01', '3_07_02', '3_07_03', '3_07_04', '3_07_05'],
    '3_08': ['3_08_01', '3_08_02'],
    '3_09': ['3_09_01', '3_09_02', '3_09_03', '3_09_04', '3_09_07', '3_09_08']}

model_paths = os.path.join('LearningAgents', 'saved_model')

for model_path in os.listdir(model_paths):
    if 'capability' in model_path:
        print('testing {}'.format(model_path))
        param_name = model_path[:-3]
        template_name = "_".join(param_name.split("_")[1:3])
        game_version = param_name.split('_')[-2]
        level_path = param_name.split('_')[-4]+ '_' + param_name.split('_')[-3]
        mode = param_name.split("_")[-5]
        if mode == 'train1testrest':
            train_template = capability_templates_dict[template_name][0]
            test_template = capability_templates_dict[template_name][1:]

        elif mode == 'trainhalftesthalf':
            num_temp = len(capability_templates_dict[template_name])
            train_template = capability_templates_dict[template_name][:num_temp // 2 + 1]
            test_template = capability_templates_dict[template_name][num_temp // 2 + 1:]

        elif mode == 'trainresttestone':
            train_template = capability_templates_dict[template_name][1:]
            test_template = capability_templates_dict[template_name][0]

        elif mode == 'benchmark':
            train_template = benchmark_capability_templates_dict[template_name]['train']
            test_template = benchmark_capability_templates_dict[template_name]['test']

        else:
            raise NotImplementedError('unknown mode {}'.format(mode))

        param = Parameters(template=train_template, test_template=test_template, level_path=level_path,
                           game_version=game_version)
        c = config(**param.param)
        param_name = param_name + "_offline"
        # check if file exists
        if os.path.exists('final_run/{}'.format(param_name)):
            print('{} exists, skipping.. if you want to test it again, please delete the folder in final_run'.format(
                param_name))
            continue
        writer = SummaryWriter(log_dir='final_run/{}'.format(param_name), comment=param_name)

        network = c.network(c.h, c.w, c.output, writer, c.device).to(c.device)
        network.load_state_dict(torch.load(os.path.join(model_paths, model_path), map_location='cuda:0'))
        network.eval()
        network.if_save_local = False
        network.to('cuda:0')
        network.device = 'cuda:0'
        level_winning_rate = {i: 0 for i in c.test_level_list}
        memory = c.memory_type(c.memory_size)
        episodic_rewards = []
        winning_rates = []

        agents = []
        for i in range(c.num_worker):
            level_sampled = sample_levels(c.test_level_list, c.num_worker, i)
            if not level_sampled:
                continue
            env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed, game_version=game_version)
            agent = c.agent(id=i + 1, network=network, level_list=level_sampled, replay_memory=memory, env=env,
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
