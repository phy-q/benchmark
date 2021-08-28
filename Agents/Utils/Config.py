import os
from shutil import copyfile

import lxml.etree as etree


class config:

    def __init__(self, os=None, device=None, h=None, w=None, output=None, num_update_steps=None,
                 num_level_per_agent=None, num_worker=None, multiagent=None, training_attempts_per_level=None,
                 memory_size=None, memory_type=None, singleagent=None, lr=None, train_time_per_ep=None,
                 train_time_rise=None, train_batch=None, gamma=None, eps_start=None, eps_test=None, network=None,
                 reward_type=None, eval_freq=None, train_template=None, test_template=None, simulation_speed=None,
                 online_training=None, test_steps=None):
        # operating system
        self.test_steps = test_steps
        self.online_training = online_training
        self.os = os

        # pytorch parameters
        self.device = device

        # image network parameters
        self.h = h
        self.w = w
        self.output = output

        # multiagent trainning parameters
        self.num_update_steps = num_update_steps
        self.num_level_per_agent = num_level_per_agent
        self.num_worker = num_worker
        self.multiagent = multiagent
        self.training_attempts_per_level = training_attempts_per_level
        self.memory_size = memory_size
        self.memory_type = memory_type

        # single agent training parameters
        self.singleagent = singleagent

        # general trainning parameters

        self.lr = lr
        self.train_time_per_ep = train_time_per_ep
        self.train_time_rise = train_time_rise
        self.train_batch = train_batch
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_test = eps_test  # randomeness in testing
        self.network = network
        self.reward_type = reward_type  # 'score' / 'passing'
        self.simulation_speed = simulation_speed
        self.eval_freq = eval_freq

        self.train_template = train_template  # ['1_1_1', '1_1_2'] level 1 capability 1 template 1 and 2 for training
        self.test_template = test_template  # ['1_1_3', '1_1_4'] level 1 capability 1 template 3 and 4 for testing

        self.train_level_list = []
        self.test_level_list = []
        self.total_level = 0
        self.target_level_path = '../tasks/generated_tasks/'
        self.origin_level_path = '../buildgame/{}/9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels/'.format(
            self.os)
        self.game_level_path = '9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels/'.format(self.os)
        self.game_config_path = '../buildgame/{}/config.xml'.format(self.os)
        self.update_level_index()

    def update_level_index(self):
        '''
        by taking the TRAIN_TEMPLATE and TEST_TEMPLATE,
        this function create the config list for training and testing

        :return:
        '''

        # remove all the levels in self.origin_level_path
        old_levels = os.listdir(self.origin_level_path)
        for old_level in old_levels:
            os.remove(os.path.join(self.origin_level_path, old_level))

        if len(self.train_template) == 1 and self.train_template == self.test_template:  # within template training and testing
            # by default using 80% of levels to train and 20% for testing
            train_percent = 0.8
            level, capability, template_idx = self.train_template[0].split('_')
            template_path = os.path.join(self.target_level_path, level, capability, template_idx)

            # copy all target levels to self.origin_level_path
            new_levels = sorted(os.listdir(template_path), key=lambda x: int(x.split('.')[0].split("_")[-1]))
            total_template_level_path = []
            for new_level in new_levels:
                src_path = os.path.join(template_path, new_level)
                dst_path = os.path.join(self.origin_level_path, new_level)
                copyfile(src_path, dst_path)
                total_template_level_path.append(os.path.join(self.game_level_path, new_level))
            self.train_level_list = list(range(1, int(len(total_template_level_path) * train_percent) + 1))
            self.test_level_list = list(
                range(int(len(total_template_level_path) * train_percent) + 1, len(total_template_level_path) + 1))

            # write configs
            parser = etree.XMLParser(encoding='UTF-8')
            game_config = etree.parse(self.game_config_path, parser=parser)
            config_root = game_config.getroot()
            # remove old level path
            for level in list(config_root[1][0][0]):
                config_root[1][0][0].remove(level)
            # add new level path
            for l in total_template_level_path:
                new_level = etree.SubElement(config_root[1][0][0], 'game_levels')
                new_level.set('level_path', l)

            # add a repeated level for the weird not loadding last level bug
            new_level = etree.SubElement(config_root[1][0][0], 'game_levels')
            new_level.set('level_path', l)

            game_config.write(self.game_config_path)

        else:  # when more than one template is required to train
            # cross template testing
            total_template_level_path = []

            for template in self.train_template:
                level, capability, template_idx = template.split('_')
                template_path = os.path.join(self.target_level_path, level, capability, template_idx)

                # move all levels to folder
                new_levels = sorted(os.listdir(template_path), key=lambda x: int(x.split('.')[0].split("_")[-1]))
                for new_level in new_levels:
                    src_path = os.path.join(template_path, new_level)
                    dst_path = os.path.join(self.origin_level_path, new_level)
                    copyfile(src_path, dst_path)
                    total_template_level_path.append(os.path.join(self.game_level_path, new_level))

            self.train_level_list = [i for i in range(1, len(total_template_level_path) + 1)]

            for template in self.test_template:
                level, capability, template_idx = template.split('_')
                template_path = os.path.join(self.target_level_path, level, capability, template_idx)

                # move all levels to folder
                new_levels = sorted(os.listdir(template_path), key=lambda x: int(x.split('.')[0].split("_")[-1]))

                # takes only the last 20 levels per each template for testing
                new_levels = new_levels[80:]

                for new_level in new_levels:
                    src_path = os.path.join(template_path, new_level)
                    dst_path = os.path.join(self.origin_level_path, new_level)
                    copyfile(src_path, dst_path)
                    total_template_level_path.append(os.path.join(self.game_level_path, new_level))

            self.test_level_list = [i for i in range(self.train_level_list[-1] + 1, len(total_template_level_path) + 1)]

            # write config
            # write configs
            parser = etree.XMLParser(encoding='UTF-8')
            game_config = etree.parse(self.game_config_path, parser=parser)
            config_root = game_config.getroot()
            # remove old level path
            for level in list(config_root[1][0][0]):
                config_root[1][0][0].remove(level)
            # add new level path
            for l in total_template_level_path:
                new_level = etree.SubElement(config_root[1][0][0], 'game_levels')
                new_level.set('level_path', l)

            # add a repeated level for the weird not loadding last level bug
            new_level = etree.SubElement(config_root[1][0][0], 'game_levels')
            new_level.set('level_path', l)

            game_config.write(self.game_config_path)


if __name__ == '__main__':
    from LearningAgents.DQNImageAgent import DQNImageAgent
    from LearningAgents.RL.DQNImageDueling import DQNImageDueling

    test_config = {
        # operating system
        'os': "Linux",
        # pytorch parameters
        'device': "cuda:0",

        # image network parameters
        'h': 224,
        'w': 224,
        'output': 180,

        # multiagent trainning parameters
        'num_update_steps': 100,
        'num_level_per_agent': 20,
        'num_worker': 30,
        'multiagent': DQNImageAgent,

        # single agent training parameters
        'singleagent': None,

        # general trainning parameters

        'train_time_per_ep': 32,
        'train_time_rise': 1.5,
        'train_batch': 32,
        'gamma': 0.99,
        'eps_start': 0.95,
        'eps_decay': 0.99,

        'network': DQNImageDueling,

        'train_template': ['1_1_1'],  # ['1_1_1', '1_1_2'] level 1 capability 1 template 1 and 2 for training
        'test_template': ['1_1_3'],  # ['1_1_3', '1_1_4'] level 1 capability 1 template 3 and 4 for testing
    }
    c = config(**test_config)

    print(c.train_level_list)
    print(c.test_level_list)
