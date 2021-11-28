import math
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from Utils.LevelSelection import LevelSelectionSchema

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

model_paths = os.path.join('LearningAgents', 'saved_model')

for model_path in os.listdir(model_paths):
    print('testing {}'.format(model_path))
    if 'capability' in model_path:
        continue
    param_name = model_path[:-3]
    template_name = "_".join(param_name.split("_")[:3])
    game_version = param_name.split('_')[-2]
    level_path = param_name.split('_')[-4]+ '_' + param_name.split('_')[-3]
    param = Parameters([template_name], level_path=level_path, game_version=game_version)
    c = config(**param.param)
    param_name = param_name + "_offline"

    # check if file exists
    if os.path.exists('final_run/{}'.format(param_name)):
        print('{} exists, skipping.. if you want to test it again, please delete the folder in final_run'.format(
            param_name))
        continue

    writer = SummaryWriter(log_dir='final_run/{}'.format(param_name), comment=param_name)

    network = c.network(c.h, c.w, c.output, writer, c.device).to(c.device)
    network.load_state_dict(torch.load(os.path.join(model_paths,model_path), map_location='cuda:0'))
    network.eval()
    network.if_save_local = False
    network.to('cuda:0')
    network.device = 'cuda:0'
    memory = c.memory_type(c.memory_size)
    episodic_rewards = []
    winning_rates = []
    level_winning_rate = {i: 0 for i in c.test_level_list}

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
            except KeyError:  # agent skipped level
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
    del agents
    torch.cuda.empty_cache()

