import argparse
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from LearningAgents.RLNetwork.OpenAICustomCNN import OpenAICustomCNN
from SBEnvironment.SBEnvironmentWrapperOpenAI import SBEnvironmentWrapperOpenAI
from SBEnvironment.Server import Server
from Utils.Config import config
from Utils.Parameters import Parameters
from Utils.utils import make_env, sample_levels_with_at_least_num_agents

warnings.filterwarnings('ignore')

# Set a seed value
seed_value = 5123690
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

logger = logging.getLogger("OpenAI stable baselines Training and Testing")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, rollout_buffer, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.rollout_buffer = rollout_buffer

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record('train/average episodic reward', np.mean(self.rollout_buffer.rewards))
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', metavar='N', type=str)
    parser.add_argument('--mode',
                        type=str,
                        default='within_template')  # propose three modes, 'train1testrest', 'trainhalftesthalf', 'trainresttestone'
    parser.add_argument('--level_path', type=str, default='fifth_generation')
    parser.add_argument('--game_version', type=str, default='Linux')

    args = parser.parse_args()

    if len(args.template.split("_")) == 3:
        param = Parameters([args.template], level_path=args.level_path,
                           game_version=args.game_version)
        c = config(**param.param)
        param_name = args.template

    elif len(args.template.split("_")) == 2:
        if args.mode == 'trainonetestrest':
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

    param_name = param_name + "_" + c.action_type + "_" + c.state_repr_type + '_' + \
                 c.agent + "_" + args.mode + "_" + args.level_path + "_" + args.game_version

    logger.info('running {} mode template {} on {}'.format(args.mode, args.template, param_name))

    game_server = Server(state_repr_type=c.state_repr_type, game_version=c.game_version,
                         if_head=False if c.state_repr_type == 'symbolic' else True)
    game_server.close()

    env = SubprocVecEnv([make_env(env_id=env_id,
                                  level_list=[level_ind],
                                  action_type=c.action_type,
                                  max_attempts_per_level=1,
                                  if_init=False,
                                  state_repr_type=c.state_repr_type) for env_id, level_ind in
                         zip(range(c.num_worker),
                             range(1, c.num_worker + 1))])

    if c.state_repr_type == 'symbolic':
        policy_kwargs = dict(
            features_extractor_class=OpenAICustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
    elif c.state_repr_type == 'image':
        policy_kwargs = dict(
            normalize_images=True
        )
    else:
        raise NotImplementedError('state_repr_type {} not implemented'.format(args.state_repr_type))

    if c.agent.lower() == 'ppo':
        n_steps = c.training_attempts_per_level * (len(c.train_level_list) // c.num_worker)
        # batch_size = min(n_steps * c.num_worker // 4, 64)
        model = PPO(policy='CnnPolicy', device=c.device, batch_size=c.train_batch,
                    # n_steps is the number of step before training
                    # for template training, set up attempts * 80 // number of agents
                    # so this considers the total value
                    n_steps=c.training_attempts_per_level * (len(c.train_level_list) // c.num_worker),
                    n_epochs=c.train_time_per_ep, learning_rate=c.lr, verbose=True,
                    tensorboard_log='OpenAIStableBaseline', env=env, policy_kwargs=policy_kwargs, )
    elif c.agent.lower() == 'a2c':
        # for a2c , it seems like the n_step here means only for one agent
        n_steps = c.training_attempts_per_level  # * len(c.train_level_list) // c.num_worker
        model = A2C(policy='CnnPolicy', device=c.device, verbose=True,
                    tensorboard_log='OpenAIStableBaseline', env=env, learning_rate=c.lr,
                    policy_kwargs=policy_kwargs, )

    else:
        raise NotImplementedError("{} is not implemented".format(c.agent.lower()))

    test_writer = SummaryWriter(log_dir='OpenAIStableBaseline/{}_test_result'.format(param_name))

    # # load model #
    # if c.resume:
    #     model_path = os.path.join("OpenAIModelCheckpoints")
    #     saved_steps = []
    #     for model_name in os.listdir(model_path):
    #         if param_name in model_name:
    #             saved_steps.append(int(model_name[:-3].split("_")[-1]))
    #     max_steps = max(saved_steps)
    #     model_to_load = os.path.join(model_path, param_name + "_" + str(max_steps) + ".pt")
    #     model.load(model_to_load, device=c.device)
    #     logger.info('{} loaded'.format(model_to_load))
    # ################
    c.num_update_steps = len(c.train_level_list) if len(c.train_level_list) >= 5 else 50
    for step in range(c.num_update_steps):
        logger.info("training step: {}".format(step))
        game_server.start()

        training_level_lists = []

        for i in range(c.num_worker):
            level_sampled = sample_levels_with_at_least_num_agents(c.train_level_list, c.num_worker, i,
                                                                   level_per_agent=c.num_level_per_agent)
            training_level_lists.append(level_sampled)

        env = SubprocVecEnv([make_env(env_id=env_id,
                                      level_list=training_level_lists[env_id],
                                      action_type=c.action_type,
                                      state_repr_type=c.state_repr_type,
                                      max_attempts_per_level=c.training_attempts_per_level) for env_id in
                            range(c.num_worker)])
        time.sleep(c.num_worker * 3)
        model.set_env(env)
        if step != 0:
            env.reset()

        # for ppo the agent run the n_step * (number of agent + 1) as the n_steps for ppo is for all agents
        # for a2c, this becomes n_steps * len(training level) * (number of agent + 1)
        model.learn(
            model.n_steps * len(c.train_level_list) * c.num_worker if c.agent == 'a2c' else model.n_steps *
                                                                                                  c.num_worker,
            log_interval=1,
            callback=TensorboardCallback(model.rollout_buffer),
            tb_log_name=param_name, reset_num_timesteps=False)
        game_server.close()
        time.sleep(5)
    # last test
    score = []
    winning_rate = []
    game_server.start()

    env = SBEnvironmentWrapperOpenAI(0, c.test_level_list, c.action_type, c.state_repr_type,
                                     max_attempts_per_level=1)

    obs = env.reset()
    level_count = 0
    while level_count < len(c.test_level_list):
        action, _states = model.predict(obs, deterministic=True)
        _, reward, done, info = env.step(action)
        print('playing {}'.format(env.current_level))
        if done:
            level_count += 1
            obs = env.reset()
            winning_rate.append(info['did_win'])
            score.append(info['score'])
    test_writer.add_scalar("test/average testing score", np.average(score), c.num_update_steps)
    test_writer.add_scalar("test/average testing passing rate", np.average(winning_rate), c.num_update_steps)

    game_server.close()
    model.save(os.path.join("OpenAIModelCheckpoints", param_name + "_" + str(c.num_update_steps)))
