
import argparse
import math
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from Utils.LevelSelection import LevelSelectionSchema
from HeuristicAgents.HeuristicAgentThread import MultiThreadTrajCollection
from HeuristicAgents.PigShooter import PigShooter
from HeuristicAgents.RandomAgent import RandomAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.Parameters import Parameters


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', metavar='N', type=str)
    parser.add_argument('--agent', type=str, default='PigShooter')
    parser.add_argument('--level_path', type=str, default='fifth_generation')
    parser.add_argument('--game_version', type=str, default='Linux')
    parser.add_argument('--if_save_local', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    args = parser.parse_args()

    if args.agent == 'PigShooter':
        test_agent = PigShooter
    elif args.agent == 'RandomAgent':
        test_agent = RandomAgent
    else:
        raise NotImplementedError(
            "heuristic agent {} not implemented. Please use PigShooter or RandomAgent".format(args.agent))

    
    param = Parameters([args.template], level_path=args.level_path,
                           game_version=args.game_version)
    c = config(**param.param)
    param_name = args.template + "_random"
    writer = SummaryWriter(log_dir='final_run/{}'.format(param_name), comment=param_name)

    #####################################running phase#########################################
    test_attempts_per_level = 1
    level_winning_rate = {i: 0 for i in c.test_level_list}

    for attempt in range(50):
        agents = []
        for i in range(c.num_worker):
            level_sampled = sample_levels(c.test_level_list, c.num_worker,
                                          i)  # test level set needs to be the 1640 levels
            if not level_sampled:
                continue
            env = SBEnvironmentWrapper(reward_type='passing', speed=100)
            agent = test_agent(env=env, level_list=level_sampled, id=i + 1,
                               level_selection_function=LevelSelectionSchema.RepeatPlay(
                                   test_attempts_per_level).select)  # add number of attempts per level
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
                          (attempt + 1) * test_attempts_per_level)
        writer.add_scalar("average_testing_winning_rates", np.average(winning_rate),
                          (attempt + 1) * test_attempts_per_level)
        writer.add_scalar("max_testing_episodic_rewards", np.average(max_reward),
                          (attempt + 1) * test_attempts_per_level)
        writer.add_scalar(
            "max_testing_winning_rates - level is solved",
            np.average(max_winning_rate), (attempt + 1) * test_attempts_per_level)
        # percent of task solved
        percent_task_solved = np.average(list(level_winning_rate.values()))
        writer.add_scalar("percent of testing tasks solved",
                          percent_task_solved, (attempt + 1) * test_attempts_per_level)
        writer.flush()
        # del model and agents to free memory
        # del dqn
        del agents
