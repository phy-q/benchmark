import os
from shutil import copyfile

import lxml.etree as etree


class config:

    def __init__(self, **kwargs):
        self.resume = kwargs['resume'] if 'resume' in kwargs else None
        self.action_type = kwargs['action_type'] if 'action_type' in kwargs else None
        self.state_repr_type = kwargs['state_repr_type'] if 'state_repr_type' in kwargs else None
        # operating system
        self.test_steps = kwargs['test_steps'] if 'test_steps' in kwargs else None
        self.os = kwargs['os'] if 'os' in kwargs else None

        # pytorch parameters
        self.device = kwargs['device'] if 'device' in kwargs else None

        # image network parameters
        self.h = kwargs['h'] if 'h' in kwargs else None
        self.w = kwargs['w'] if 'w' in kwargs else None
        self.output = kwargs['output'] if 'output' in kwargs else None

        # multiagent trainning parameters
        self.num_update_steps = kwargs['num_update_steps'] if 'num_update_steps' in kwargs else None
        self.num_level_per_agent = kwargs['num_level_per_agent'] if 'num_level_per_agent' in kwargs else None
        self.num_worker = kwargs['num_worker'] if 'num_worker' in kwargs else None
        self.agent = kwargs['agent'] if 'agent' in kwargs else None
        self.training_attempts_per_level = kwargs['training_attempts_per_level'] if 'training_attempts_per_level' in kwargs else None
        self.memory_size = kwargs['memory_size'] if 'memory_size' in kwargs else None
        self.memory_type = kwargs['memory_type'] if 'memory_type' in kwargs else None

        # general trainning parameters

        self.lr = kwargs['lr'] if 'lr' in kwargs else None
        self.train_time_per_ep = kwargs['train_time_per_ep'] if 'train_time_per_ep' in kwargs else None
        self.train_time_rise = kwargs['train_time_rise'] if 'train_time_rise' in kwargs else None
        self.train_batch = kwargs['train_batch'] if 'train_batch' in kwargs else None
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else None
        self.eps_start = kwargs['eps_start'] if 'eps_start' in kwargs else None
        self.eps_test = kwargs['eps_test'] if 'eps_test' in kwargs else None  # randomeness in testing
        self.network = kwargs['network'] if 'network' in kwargs else None
        self.reward_type = kwargs['reward_type'] if 'reward_type' in kwargs else None  # 'score' / 'passing'
        self.simulation_speed = kwargs['simulation_speed'] if 'simulation_speed' in kwargs else None
        self.eval_freq = kwargs['eval_freq'] if 'eval_freq' in kwargs else None
        self.game_version = kwargs['game_version'] if 'game_version' in kwargs else None

        self.train_template = kwargs['train_template'] if 'train_template' in kwargs else None  # ['1_1_1', '1_1_2'] level 1 capability 1 template 1 and 2 for training
        self.test_template = kwargs['test_template'] if 'test_template' in kwargs else None  # ['1_1_3', '1_1_4'] level 1 capability 1 template 3 and 4 for testing
        self.level_path = kwargs['level_path'] if 'level_path' in kwargs else None

        self.train_level_list = []
        self.test_level_list = []
        self.total_level = 0
        self.target_level_path = '../sciencebirdslevels/generated_levels/{}'.format(self.level_path)
        self.origin_level_path = '../sciencebirdsgames/{}/9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels/'.format(
            self.game_version)
        self.game_level_path = '9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels/'.format(self.os)
        self.game_config_path = '../sciencebirdsgames/{}/config.xml'.format(self.game_version)
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
            try:
                new_levels = sorted(os.listdir(template_path), key=lambda x: int(x.split('.')[0].split("_")[-1]))
            except ValueError:
                new_levels = sorted(os.listdir(template_path), key=lambda x: int(x.split('.')[0].split("-")[-1]))

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
