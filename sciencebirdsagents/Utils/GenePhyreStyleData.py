import argparse
import math
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from HeuristicAgents.CollectionAgentThread import MultiThreadTrajCollection
from HeuristicAgents.RandomAgent import RandomAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.LevelSelection import LevelSelectionSchema
from    Utils.Parameters import Parameters

input_w = 128
input_h = 128
mask_size = 21
max_p_acts = 100
max_n_acts = 400

one_shot_template = [
    # '1_01_01','1_01_02', '1_01_04','1_01_05', '1_01_06',
    # '2_01_02','2_01_03', '2_01_05', '2_01_06', '2_01_08',
    # '2_01_09', '2_02_01', '2_02_02', '2_02_03', '2_02_06',
    # '2_02_08', '2_03_01','2_03_03','2_03_04', '2_03_05',
    # '2_03_02', '2_04_01','2_04_02', '2_04_04', '2_04_03',
    '2_04_05', '2_04_06', '3_01_01', '3_01_02','3_01_03',
    '3_01_04', '3_01_06', '3_02_01', '3_02_02', '3_02_03', '3_02_04', '3_03_01', '3_03_02', '3_03_03', '3_03_04', '3_04_01',
    '3_04_02', '3_04_03', '3_04_04', '3_06_01', '3_06_03', '3_06_04', '3_06_05', '3_06_06',
    '3_06_07',
]


def arg_parse():
    parser = argparse.ArgumentParser(description='PhyreStyleData parameters')
    parser.add_argument('--range', type=str, default='10,170',
                        help='min angle to max angle, e.g. 10,170 meaning shooting from degree 10 to degree 170')
    parser.add_argument('--num_shot', type=int, default=100, help='number of shots per game level')
    parser.add_argument('--num_worker', type=int, default=10, help='number of worke to run')
    parser.add_argument('--headless_server', type=int, default=0, help='0 for running in graphic mode, 1 for command line')

    return parser.parse_args()


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
    args = arg_parse()

    for template in tqdm(one_shot_template):
        param = Parameters([template], test_template=['1_02_01'], level_path='fifth_generation',
                           game_version='Linux')
        c = config(**param.param)
        agents = []
        for i in range(args.num_worker):
            level_sampled = sample_levels(c.train_level_list, args.num_worker, i)
            if not level_sampled:
                continue
            env = SBEnvironmentWrapper(reward_type='passing', speed=100, headless_server=True if args.headless_server==1 else False)
            agent = RandomAgent(env=env, level_list=level_sampled, id=i + 1,
                                level_selection_function=LevelSelectionSchema.RepeatPlay(
                                    args.num_shot).select, degree_range=[int(args.range.split(',')[0]),int(args.range.split(',')[1])])  # add number of attempts per level
            agent.template = template
            agents.append(agent)

        am = MultiThreadTrajCollection(agents)
        am.connect_and_run_agents()
        env.close()
