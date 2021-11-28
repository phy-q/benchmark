import argparse
import math
import random

from SBEnvironment.SBEnvironmentWrapperOpenAI import SBEnvironmentWrapperOpenAI


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


def sample_levels_with_at_least_num_agents(training_level_set, num_agents, agent_idx, **kwargs):
    '''
    given idx, return the averaged distributed levels
    '''

    level_per_agent = kwargs['level_per_agent'] if 'level_per_agent' in kwargs else None
    total_list = []
    for _ in range(num_agents):
        total_list.append([])
    i = 0
    while i < len(training_level_set):
        for level_list in total_list:
            level_list.append(training_level_set[i])
            i += 1

    if level_per_agent:
        if level_per_agent < len(total_list[agent_idx]):
            levels = random.sample(total_list[agent_idx], level_per_agent)
        else:
            levels = total_list[agent_idx]
    return levels


def make_env(env_id, level_list, action_type, state_repr_type, max_attempts_per_level, if_init=True):
    def _init():
        env = SBEnvironmentWrapperOpenAI(env_id, level_list, action_type, state_repr_type, if_init=if_init,
                                         max_attempts_per_level=max_attempts_per_level)
        return env

    return _init
