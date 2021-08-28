import argparse
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from LearningAgents.LearningAgentThread import MultiThreadTrajCollection
from SBEnviornment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.LevelSelection import LevelSelectionSchema
from Utils.Parameters import Parameters

warnings.filterwarnings('ignore')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sample_levels(training_level_set, num_agents, agent_idx, **kwargs):
    '''
    given idx, return the averaged distributed levels
    '''
    level_per_agent = kwargs['level_per_agent'] if 'level_per_agent' in kwargs else None
    n = math.ceil(len(training_level_set) / num_agents)
    total_list = []
    for i in range(0, len(training_level_set), n):
        levels_assigned = training_level_set[i:i + n]
        if level_per_agent:
            if n > level_per_agent:
                levels_assigned = random.sample(levels_assigned, level_per_agent)
        total_list.append(sorted(levels_assigned))
    if agent_idx >= len(total_list):
        return None
    return total_list[agent_idx]


capabiilty_settings = {'1_01': {'train': ['1_01_01', '1_01_02'], 'test': ['1_01_03']},
                       '1_02': {'train': ['1_02_01', '1_02_02', '1_02_03'], 'test': ['1_02_04', '1_02_05']},
                       '2_01': {'train': ['2_01_01', '2_01_02', '2_01_03'],
                                'test': ['2_01_04', '2_01_05']},
                       '2_02': {'train': ['2_02_01', '2_02_02', '2_02_03'],
                                'test': ['2_02_04', '2_02_05']},
                       '2_03': {'train': ['2_03_01', '2_03_02'], 'test': ['2_03_03', '2_03_04']},
                       '2_04': {'train': ['2_04_01', '2_04_03'], 'test': ['2_04_02']},
                       '3_01': {'train': ['3_01_01', '3_01_02', '3_01_03'], 'test': ['3_01_04', '3_01_05']},
                       '3_02': {'train': ['3_02_01', '3_02_02'], 'test': ['3_02_03', '3_02_04']},
                       '3_03': {'train': ['3_03_01', '3_03_02'], 'test': ['3_03_03', '3_03_04']},
                       '3_04': {'train': ['3_04_01', '3_04_02'], 'test': ['3_04_03', '3_04_04']},
                       '3_05': {'train': ['3_05_01', '3_05_03', '3_05_04'], 'test': ['3_05_02', '3_05_05']},
                       '3_06': {'train': ['3_06_01', '3_06_03', '3_06_05'],
                                'test': ['3_06_02', '3_06_04']},
                       '3_07': {'train': ['3_07_01', '3_07_02', '3_07_03'], 'test': ['3_07_04', '3_07_05']},
                       '3_08': {'train': ['3_08_01'], 'test': ['3_08_02']},
                       '3_09': {'train': ['3_09_01', '3_09_02', '3_09_03', '3_09_04'],
                                'test': ['3_09_05', '3_09_06']}}

level_1_template = []
level_2_template = []
level_3_template = []
for cap in capabiilty_settings:
    train_list = capabiilty_settings[cap]['train']
    test_list = capabiilty_settings[cap]['test']
    if cap[0] == '1':
        level_1_template.extend(train_list + test_list)
    elif cap[0] == '2':
        level_2_template.extend(train_list + test_list)
    elif cap[0] == '3':
        level_3_template.extend(train_list + test_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    param = Parameters(level_1_template + level_2_template + level_3_template, False)
    param1 = Parameters(template=level_1_template, if_online_learning=False)
    param1_2 = Parameters(template=level_1_template + level_2_template, if_online_learning=False)
    param3 = Parameters(template=level_3_template, if_online_learning=False)
    param3_2 = Parameters(template=level_3_template + level_2_template, if_online_learning=False)

    if args.mode == 'no_curriculum':
        param_name = "no_curriculum"
        param_list = [param.param]

    elif args.mode == 'curriculum':
        param_name = 'curriculum'
        param_list = [param1.param, param1_2.param, param.param]

    elif args.mode == 'reverse_curriculum':
        param_name = 'reverse_curriculum'
        param_list = [param3.param, param3_2.param, param.param]

    else:
        raise NotImplementedError(
            "unknown mode {} for curriculum learning, please use curriculum, reverse_curriculum or no_curriculum".format(
                args.mode))

    print('running {}'.format(args.mode))
    if not os.path.exists('final_run'):
        os.mkdir('final_run')

    writer = SummaryWriter(log_dir='final_run/{}'.format(param_name), comment=param_name)
    network = param_list[0]['network'](param_list[0]['h'], param_list[0]['w'], param_list[0]['output'], writer,
                                     param_list[0]['device']).to(param_list[0]['device'])
    memory = param_list[0]['memory_type'](param_list[0]['memory_size'])
    optimizer = optim.Adam

    episodic_rewards = []
    winning_rates = []
    total_train_time = 0
    total_update_steps = len(level_1_template + level_2_template + level_3_template) * 5
    eps_decay = (0.99 / (total_update_steps - 20))
    level_winning_rate = {i: 0 for i in config(**param.param).train_level_list}
    start_time = time.time()
    update_steps = 0

    for p in param_list:
        if len(param_list) == 1:
            num_update_steps = total_update_steps
        else:
            num_update_steps = total_update_steps // 3
        #####################################training phase#########################################
        c = config(**p)
        for step in range(num_update_steps):
            c.eps_start = c.eps_start - eps_decay
            ## using multi-threading to collect memory ##
            agents = []
            for i in range(c.num_worker):
                level_sampled = sample_levels(c.train_level_list, c.num_worker, i,
                                              level_per_agent=c.num_level_per_agent if step != 0 else None)
                env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed)
                agent = c.multiagent(id=i + 1, dqn=network, level_list=level_sampled, replay_memory=memory, env=env,
                                     level_selection_function=LevelSelectionSchema.RepeatPlay(
                                         c.training_attempts_per_level).select,
                                     EPS_START=c.eps_start, writer=writer)
                agents.append(agent)

            am = MultiThreadTrajCollection(agents)
            am.connect_and_run_agents()
            env.close()
            time.sleep(5)

            ## evaluate the agent's learning performance ##
            episodic_reward = []
            winning_rate = []
            max_reward = []
            max_winning_rate = []

            for idx in c.train_level_list:
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
                    except KeyError:  # agent skipped level
                        episodic_reward.append(0)
                        winning_rate.append(0)

            writer.add_scalar("average_episodic_rewards", np.average(episodic_reward), memory.action_num)
            writer.add_scalar("average_winning_rates", np.average(winning_rate), memory.action_num)
            writer.add_scalar("max_episodic_rewards", np.average(max_reward), memory.action_num)
            writer.add_scalar(
                "max_winning_rates - training level is solved per agent",
                np.average(max_winning_rate), memory.action_num)
            # percent of task solved
            percent_task_solved = np.average(list(level_winning_rate.values()))
            writer.add_scalar("percent of training tasks solved",
                              percent_task_solved, memory.action_num)
            # del model and agents to free memory
            # del dqn
            del agents
            torch.cuda.empty_cache()

            ## training the network ##
            target_net = c.network(h=c.h, w=c.w, outputs=c.output, device=c.device).to(c.device)
            target_net.load_state_dict(network.state_dict())
            # target_net.eval()
            network.train_model(target_net, total_train_time=total_train_time,
                                train_time=len(memory) // c.train_batch * 10,
                                train_batch=c.train_batch, gamma=c.gamma, memory=memory, optimizer=optimizer,
                                sample_eps=c.eps_start)

            del target_net
            torch.cuda.empty_cache()
            print('finish {} step'.format(step))
            end_time = time.time()
            print("running time: {:.2f}".format((end_time - start_time) / 60))
            total_train_time += c.train_time_per_ep

        print('one level training done')
    # training done, save the model
    if not os.path.exists('LearningAgents/saved_model'):
        os.mkdir('LearningAgents/saved_model')
    network.save_model("LearningAgents/saved_model/{}_{}.pt".format(param_name, 'last'))
    total_end_time = time.time()
    print("finish running, total running time: {:.2f} mins".format((total_end_time - start_time) / 60))
