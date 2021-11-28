import argparse
import logging
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from LearningAgents.LearningAgentThread import MultiThreadTrajCollection
from LearningAgents.MemoryDataset import MemoryDatasetMemmap
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.LevelSelection import LevelSelectionSchema
from Utils.Parameters import Parameters

warnings.filterwarnings('ignore')

# Set a seed value
seed_value = 5123690
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

logger = logging.getLogger("Main Training and Testing")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# torch.use_deterministic_algorithms(True)


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

    n = max(math.ceil(len(training_level_set) / num_agents), 1)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', metavar='N', type=str)
    parser.add_argument('--mode',
                        type=str)  # propose three modes, 'train1testrest', 'trainhalftesthalf', 'trainresttestone', 'benchmark'
    parser.add_argument('--level_path', type=str, default='fourth generation')
    parser.add_argument('--game_version', type=str, default='Linux')
    parser.add_argument('--if_save_local', type=str2bool, default=True)
    parser.add_argument('--resume', type=str2bool, default=False)

    args = parser.parse_args()

    memory_saving_path = os.path.join(Path.home(), 'TrainingMemory')
    memory_saving_path = os.path.join('LearningAgents', 'saved_memory')

    if len(args.template.split("_")) == 3:
        param = Parameters([args.template], level_path=args.level_path,
                           game_version=args.game_version)
        c = config(**param.param)
        param_name = args.template

    elif len(args.template.split("_")) == 2:
        if args.mode == 'train1testrest':
            capability_idx = args.template
            train_template = [capability_templates_dict[capability_idx][0]]
            test_template = capability_templates_dict[capability_idx][1:]
            param = Parameters(template=train_template,
                               test_template=test_template, level_path=args.level_path, game_version=args.game_version)
            c = config(**param.param)
            param_name = "capability_{}".format(train_template[0][:4])

        elif args.mode == 'trainhalftesthalf':
            capability_idx = args.template
            num_temp = len(capability_templates_dict[capability_idx])
            train_template = capability_templates_dict[capability_idx][:num_temp // 2 + 1]
            test_template = capability_templates_dict[capability_idx][num_temp // 2 + 1:]
            param = Parameters(template=train_template,
                               test_template=test_template, level_path=args.level_path, game_version=args.game_version)
            c = config(**param.param)
            param_name = "capability_{}".format(train_template[0][:4])

        elif args.mode == 'trainresttestone':
            capability_idx = args.template
            train_template = capability_templates_dict[capability_idx][1:]
            test_template = [capability_templates_dict[capability_idx][0]]
            param = Parameters(template=train_template,
                               test_template=test_template, level_path=args.level_path, game_version=args.game_version)
            c = config(**param.param)
            param_name = "capability_{}".format(train_template[0][:4])
        elif args.mode == 'benchmark':
            capability_idx = args.template
            train_template = benchmark_capability_templates_dict[capability_idx]['train']
            test_template = benchmark_capability_templates_dict[capability_idx]['test']
            param = Parameters(template=train_template,
                               test_template=test_template, level_path=args.level_path, game_version=args.game_version)
            c = config(**param.param)
            param_name = "capability_{}".format(train_template[0][:4])
        else:
            raise NotImplementedError("{} mode not implemented ".format(args.mode))
    else:
        raise NotImplementedError('{} not defined'.format(args.template))

    param_name = param_name + "_" + c.network.__name__ + "_" + c.agent.__name__ + "_" + c.memory_type.__name__ + "_" + args.mode + "_" + args.level_path + "_" + args.game_version
    logger.info('running {} mode template {} on {}'.format(args.mode, args.template, param_name))
    if not os.path.exists('final_run'):
        os.mkdir('final_run')
    writer = SummaryWriter(log_dir='final_run/{}'.format(param_name), comment=param_name)
    if c.agent.__name__ != 'SACAgent':
        network = c.network(h=c.h, w=c.w, outputs=c.output, if_save_local=args.if_save_local, writer=writer,
                            device=c.device, ).to(c.device)
    else:
        network = c.network(h=c.h, w=c.w, n_actions=3, if_save_local=args.if_save_local, writer=writer,
                            lr=param.param['lr'],
                            device=c.device, reward_scale=1)

    if args.resume:
        model_path = os.path.join("LearningAgents", "saved_model")
        saved_steps = []
        for model_name in os.listdir(model_path):
            if param_name in model_name:
                saved_steps.append(int(model_name[:-3].split("_")[-1]))
        max_steps = max(saved_steps)
        model_to_load = os.path.join(model_path, param_name + "_" + str(max_steps) + ".pt")
        network.load_state_dict(torch.load(model_to_load, map_location=param.param['device']))
        logger.info('{} loaded'.format(model_to_load))

    memory = c.memory_type(c.memory_size)
    optimizer = optim.AdamW

    episodic_rewards = []
    winning_rates = []
    total_train_time = 0
    # we want the last 1 steps to be only 0.01, so the it should be eps_start*(eps_decay)**num_update_steps = 0.01
    eps_decay = (0.99 / (c.num_update_steps - c.num_update_steps // 4))
    if len(c.train_level_list) == 0:
        c.train_level_list = [1]
    level_winning_rate = {i: 0 for i in c.train_level_list}
    start_time = time.time()

    if args.if_save_local:
        # training_set = MemoryDatasetInRam(h=c.h, w=c.w, path='LearningAgents/saved_memory')
        training_set = MemoryDatasetMemmap(h=c.h, w=c.w, c=12 if network.input_type == 'symbolic' else 3,
                                           path=memory_saving_path, logger=logger, max_len=c.memory_size)
        logger.info('dataset loaded')

    #####################################training phase#########################################
    if args.level_path != "MattewLevel":
        # for benchmark, the traing step is the number of training levels divide by 5
        c.num_update_steps = len(c.train_level_list) // 5 if len(c.train_level_list) >= 5 else 50
        eps_decay = (0.99 / (c.num_update_steps - c.num_update_steps // 10))

    for step in range(c.num_update_steps):
        logger.info("training step: {}".format(step))
        c.train_time_per_ep = max(1, c.train_time_per_ep - int(step * c.train_time_rise))  # reduce training step
        c.eps_start = max(c.eps_test, c.eps_start - eps_decay)

        ## using multi-threading to collect memory ##
        agents = []
        for i in range(c.num_worker):
            level_sampled = sample_levels(c.train_level_list, c.num_worker, i,
                                          level_per_agent=c.num_level_per_agent if step != 0 else None)
            env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed,
                                       game_version=args.game_version, if_head=False)

            if c.agent.__name__ == 'SACAgent':
                agent = c.agent(id=i + 1, level_list=level_sampled, replay_memory=memory, network=network,
                                env=env, level_selection_function=LevelSelectionSchema.RepeatPlay(
                        c.training_attempts_per_level if step != 0 else 10).select, writer=writer)
            else:
                agent = c.agent(id=i + 1, network=network, level_list=level_sampled, replay_memory=memory, env=env,
                                level_selection_function=LevelSelectionSchema.RepeatPlay(
                                    c.training_attempts_per_level if step != 0 else 10).select, EPS_START=c.eps_start,
                                writer=writer)

            agents.append(agent)

        am = MultiThreadTrajCollection(agents, memory_saving_path=memory_saving_path)
        am.connect_and_run_agents()
        env.close()
        time.sleep(5)
        # logger.info("memory size: {}".format(len(memory)))
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
                    pass

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

        # so if the average winning rate is larger than 95%, we consider all the training level are solved,
        # we can stop trainning.
        if np.average(winning_rate) > 0.99:
            break

        del agents
        torch.cuda.empty_cache()
        logger.info('finished {} step experience collection'.format(step))
        ## training the network ##
        if 'DQN' in c.network.__name__:
            target_net = c.network(h=c.h, w=c.w, outputs=c.output, device=c.device,
                                   if_save_local=args.if_save_local).to(c.device)
            target_net.load_state_dict(network.state_dict())
            target_net.eval()
        if args.if_save_local:
            sample_weights = training_set.update()
            logger.info('dataset updated')
            if not isinstance(sample_weights, type(None)):
                logger.info('dataset sample weights updated')
                train_sampler = WeightedRandomSampler(weights=sample_weights,
                                                      num_samples=len(sample_weights))
                loader = DataLoader(training_set, batch_size=c.train_batch, pin_memory=True, num_workers=0,
                                    drop_last=True, sampler=train_sampler)
            else:
                loader = DataLoader(training_set, batch_size=c.train_batch, pin_memory=True, shuffle=True,
                                    num_workers=0,
                                    drop_last=True)

            if 'DQN' in c.network.__name__:
                network.train_model_loader(target_net, total_train_time=total_train_time,
                                           train_time=c.train_time_per_ep,
                                           gamma=c.gamma, loader=loader, optimizer=optimizer, lr=param.param['lr'])
                del target_net

            else:
                network.train_model_loader(total_train_time=total_train_time, train_time=c.train_time_per_ep,
                                           loader=loader, tau=1 - np.exp(-step / 10), reward_scale=4)
            del loader
        else:
            # todo: push sac to here as well
            network.train_model_memory(target_net, total_train_time=total_train_time,
                                       train_time=c.train_time_per_ep * len(memory),
                                       train_batch=c.train_batch, gamma=c.gamma, memory=memory, optimizer=optimizer,
                                       sample_eps=c.eps_start, lr=param.param['lr'])
            del target_net

        torch.cuda.empty_cache()
        logger.info('finished {} step training'.format(step))
        end_time = time.time()
        logger.info("running time: {:.2f} mins".format((end_time - start_time) / 60))
        total_train_time += c.train_time_per_ep

        # if not os.path.exists("LearningAgents/saved_model"):
        #     os.mkdir("LearningAgents/saved_model")
        # if (step + 1) % c.eval_freq == 0:
        #     if 'DQN' in c.network.__name__:
        #         network.save_model("LearningAgents/saved_model/{}_{}.pt".format(param_name, step))
        #     else:
        #         network.save_model(param_name, step)

    logger.info('training done')
    # training done, del memory
    if not os.path.exists("LearningAgents/saved_model"):
        os.mkdir("LearningAgents/saved_model")
    if 'DQN' in c.network.__name__:
        network.save_model("LearningAgents/saved_model/{}_{}.pt".format(param_name, "last"))
    else:
        network.save_model(param_name, "last")

    if args.if_save_local:
        # delete all related memories
        saved_path = memory_saving_path
        for file in os.listdir(saved_path):
            os.remove(os.path.join(saved_path, file))

        nextstate_set_path = os.path.join(saved_path, "..", "nextstate_set.memmap")
        state_set_path = os.path.join(saved_path, "..", "state_set.memmap")
        if os.path.exists(nextstate_set_path):
            os.remove(nextstate_set_path)
        if os.path.exists(state_set_path):
            os.remove(state_set_path)

    # start testing
    logger.info('start test')
    writer = SummaryWriter(log_dir='final_run/{}'.format(param_name+"_offline"), comment=param_name+"_offline")
    network.eval()
    network.if_save_local = False
    level_winning_rate = {i: 0 for i in c.test_level_list}
    memory = c.memory_type(c.memory_size)
    episodic_rewards = []
    winning_rates = []

    agents = []
    for i in range(c.num_worker):
        level_sampled = sample_levels(c.test_level_list, c.num_worker, i)
        if not level_sampled:
            continue
        env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed, game_version=args.game_version)
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

