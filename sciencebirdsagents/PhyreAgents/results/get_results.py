import os
import argparse

import numpy as np
import pandas as pd
def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN Parameters')
    parser.add_argument('--folder', required=True, help='folder name to retrive results', type=str)

    return parser.parse_args()


def main():
    '''
    returns two csv files:
    2. scenario/template, temp_num, avg., std
    '''
    args = arg_parse()
    log_files = os.listdir(args.folder)
    ret_1 = {}

    for log_file in log_files:

        with open(os.path.join(args.folder,log_file), 'r') as f:
            temp_num = list(filter(lambda x: 'log' not in x, log_file.replace('.txt', '').split('_')))
            if len(temp_num) == 3:
                mode = 'template'
            elif len(temp_num) == 2:
                mode = 'scenario'
            else:
                raise ValueError(f'incorrect length of temp {log_file}')

            temp_num = "_".join(temp_num)

            content = f.readlines()
            passing_rates = list(filter(lambda x : 'on val levels' in x and 'INFO: 004/' in x, content))
            for rate in passing_rates:
                rate = float(rate.split(':')[-1])
                if temp_num in ret_1:
                    num_fold = len(ret_1[temp_num])
                    ret_1[temp_num][f'fold_{num_fold}'] = rate if str(rate) != 'nan' else 0
                else:
                    ret_1[temp_num] = {}
                    ret_1[temp_num]['fold_0'] = rate if str(rate) != 'nan' else 0

    out = {'template':[], 'average':[], 'std':[] }

    for temp_num in ret_1:
        avg = np.mean(list(ret_1[temp_num].values()))
        std = np.std(list(ret_1[temp_num].values()))

        out['template'].append(temp_num)
        out['average'].append(avg)
        out['std'].append(std)
    out = pd.DataFrame(out)

    out.to_csv(args.folder+'.csv', index=False)

    return out
if __name__ == '__main__':
    results = main()
