import argparse
import os
import random

import hickle
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('agg')
from utils.config import _C as cfg
from utils.logger import setup_logger
from trainer import Trainer
from models import *
from datasets import *

def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN Parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--init', type=str, help='(optionally) path to pretrained model', default='')
    parser.add_argument('--gpus', type=str, help='specification for GPU, use comma to separate GPUS', default='')
    parser.add_argument('--output', type=str, default='./', help='output name')
    parser.add_argument('--seed', type=int, help='set random seed use this command', default=0)
    parser.add_argument('--template', type=str, help='set random seed use this command', default='1_01_01')
    parser.add_argument('--protocal', type=str, help='set random seed use this command', default='template')
    parser.add_argument('--fold', type=int, help='set random seed use this command', default=0)
    parser.add_argument('--model', type=str, help='the type of model to run: dqn or rpcin', default='rpcin')

    return parser.parse_args()


def main():
    # this wrapper file contains the following procedure:
    # 1. setup training environment
    # 2. setup config
    # 3. setup logger
    # 4. setup model
    # 5. setup optimizer
    # 6. setup dataset

    # ---- setup training environment
    args = arg_parse()
    rng_seed = 0  # args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    # @\os.environ['CUDA_VISIBLE_DEVICES'] = 0 #args.gpus
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = torch.cuda.device_count()
    else:
        assert NotImplementedError

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.TEMPLATE = args.template  # cfg.TEMPLATE.replace('\\', '')
    cfg.PHYRE_PROTOCAL = args.protocal
    cfg.PHYRE_FOLD = args.fold
    cfg.SOLVER.BATCH_SIZE *= num_gpus
    cfg.SOLVER.BASE_LR *= num_gpus

    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATA_ROOT.split('/')[1], args.output)
    # os.makedirs(output_dir, exist_ok=True)
    # shutil.copy(args.cfg, os.path.join(output_dir, 'config.yaml'))
    # shutil.copy(os.path.join('RPIN/models/', cfg.ARCH + '.py'), os.path.join(output_dir, 'arch.py'))

    # ---- setup logger
    logger = setup_logger('RPIN', './output', template=cfg.TEMPLATE)
    # print(git_diff_config(args.cfg))

    # ---- setup model
    model = eval(f'{args.model}.Net')()
    model.to(torch.device('cuda'))
    # model = torch.nn.DataParallel(
    #     model, device_ids=list(range(args.gpus.count(',') + 1))
    # )

    # ---- setup optimizer
    # vae_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' in p_name]
    other_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' not in p_name]
    optim = torch.optim.Adam(
        [{'params': other_params}],
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    # if args.init:
    #     logger.info(f'loading pretrained model from {args.init}')
    #     cp = torch.load(args.init)
    #     model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    train_set = eval(f'{cfg.DATASET_ABS}')(data_root=cfg.DATA_ROOT, split='train', template=cfg.TEMPLATE,
                                           image_ext=cfg.RPIN.IMAGE_EXT)
    val_set = eval(f'{cfg.DATASET_ABS}')(data_root=cfg.DATA_ROOT, split='test', template=cfg.TEMPLATE,
                                         image_ext=cfg.RPIN.IMAGE_EXT)
    kwargs = {'pin_memory': True, 'num_workers': 16, 'drop_last': True}

    weights = {}
    data_weight = []
    for idx in train_set.video_info:
        l = hickle.load(train_set.anno_list[idx[0]].replace('boxes', 'label'))
        if l in weights:
            weights[l] += 1
        else:
            weights[l] = 1
        data_weight.append(l)

    weights[1] = weights[0] / weights[1]
    weights[0] = weights[0] / weights[0]
    data_weight = list(map(lambda x: weights[x], data_weight))

    sampler = torch.utils.data.WeightedRandomSampler(data_weight, num_samples=len(train_set))

    # load num of max objects
    import json
    num_max_obj = json.load(open('level_max_num_obj.json'))
    cfg.RPIN.MAX_NUM_OBJS = num_max_obj[args.template]
    multiplier = int(288 - (cfg.RPIN.MAX_NUM_OBJS - 6) / 15 * (288 - 82))

    cfg.SOLVER.BATCH_SIZE = multiplier // cfg.RPIN.MAX_NUM_OBJS  # based on 8gb ram of gpu

    print(cfg.RPIN.MAX_NUM_OBJS)
    print(cfg.SOLVER.BATCH_SIZE)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, sampler=sampler, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, cfg.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    cfg.SOLVER.MAX_ITERS = len(train_loader) * 5 * cfg.SOLVER.BATCH_SIZE
    cfg.SOLVER.VAL_INTERVAL = len(train_loader) // 1 * cfg.SOLVER.BATCH_SIZE
    print(cfg.SOLVER.MAX_ITERS)
    print(cfg.SOLVER.VAL_INTERVAL)
    cfg.freeze()

    # ---- setup trainer
    kwargs = {'device': torch.device('cuda'),
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'output_dir': output_dir,
              'logger': logger,
              'num_gpus': num_gpus,
              'max_iters': cfg.SOLVER.MAX_ITERS}
    trainer = Trainer(**kwargs)

    # try:
    trainer.train()
    # except BaseException:
    #    if len(glob(f"{output_dir}/*.tar")) < 1:
    #        shutil.rmtree(output_dir)
    #    raise


if __name__ == '__main__':
    main()
