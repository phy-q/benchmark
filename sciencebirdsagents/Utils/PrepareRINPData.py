import os
from tqdm import tqdm
import shutil
from glob import glob
import hickle
import numpy as np
from sklearn.model_selection import train_test_split

'''

generate
1. full: template level -> game level -> 000_image.hkl
2. images: template level -> game level -> 000.npy
3. labels: template level -> game level -> 000_boxes.hkl, 000_label.hkl, 000_masks.hkl
4. splits: protocal_[train/test]_scenario_fold_n.txt -> template level:game level

'''

output_path = './PhyreAgents/data'
input_path = './PhyreStyleTrainingData'

if not os.path.exists(output_path):
    os.mkdir(output_path)

# # prepare images folder
os.mkdir(output_path + '/full')
os.mkdir(output_path + '/images')
os.mkdir(output_path + '/labels')

for template in tqdm(os.listdir(input_path)):
    os.mkdir(output_path + '/full/' + template)
    os.mkdir(output_path + '/images/' + template)
    os.mkdir(output_path + '/labels/' + template)

    for game_level in tqdm(os.listdir(input_path + '/' + template)):
        invalid_list = []
        os.mkdir(output_path + '/full/' + template + '/' + game_level)
        os.mkdir(output_path + '/images/' + template + '/' + game_level)
        os.mkdir(output_path + '/labels/' + template + '/' + game_level)
        # move full image
        full_names = glob(f'{input_path}/{template}/{game_level}/*_image.hkl')
        for fn in full_names:
            shutil.copy(fn, f'{output_path}/full/{template}/{game_level}/{fn.split("/")[-1]}')
            # save the first
            try:
                image = hickle.load(f'{output_path}/full/{template}/{game_level}/{fn.split("/")[-1]}')[0]
            except ValueError:
                print(f'{output_path}/full/{template}/{game_level}/{fn.split("/")[-1]}')
                invalid_list.append(fn.split("/")[-1].split('_')[0])
                continue
            np.save(f'{output_path}/images/{template}/{game_level}/{fn.split("/")[-1].split("_")[0]}.npy', image)

        # move labels
        labels_names = []
        labels_names.extend(glob(f'{input_path}/{template}/{game_level}/*_boxes.hkl'))
        labels_names.extend(glob(f'{input_path}/{template}/{game_level}/*_masks.hkl'))
        labels_names.extend(glob(f'{input_path}/{template}/{game_level}/*_label.hkl'))
        for fn in labels_names:
            for inv_name in invalid_list:
                if inv_name not in fn:
                    continue

            shutil.copy(fn, f'{output_path}/labels/{template}/{game_level}/{fn.split("/")[-1]}')

# generate splits
os.mkdir(output_path + '/splits')
# within template
protocal = 'template'
total_fold = 5
test_size = 0.2
for fold in range(total_fold):
    for t in os.listdir(f'{output_path}/labels/'):
        game_levels = os.listdir(f'{output_path}/labels/{t}')
        game_levels = [f"{t}:{level}\n" for level in game_levels]
        train, test = train_test_split(game_levels, test_size=test_size)
        with open(f'{output_path}/splits/{protocal}_train_{t}_fold_{fold}.txt', 'w') as f:
            f.writelines(train)
        with open(f'{output_path}/splits/{protocal}_test_{t}_fold_{fold}.txt', 'w') as f:
            f.writelines(test)

# within scenario
protocal = 'scenario'
total_fold = 5
train_size = 0.5
for fold in range(total_fold):
    scenarios = set()
    templates = os.listdir(f'{output_path}/labels/')
    for t in templates:
        scenarios.add("_".join(t.split('_')[:2]))

    for s in list(scenarios):
        scenario_template = [t for t in templates if t[:4] == s]
        train, test = train_test_split(scenario_template, train_size=train_size)

        if len(train) < len(test):
            train, test = test, train

        for t in train:
            game_levels = os.listdir(f'{output_path}/labels/{t}')
            game_levels = [f"{t}:{level}\n" for level in game_levels]
            with open(f'{output_path}/splits/{protocal}_train_{s}_fold_{fold}.txt', 'a') as f:
                f.writelines(game_levels)
        for t in test:
            game_levels = os.listdir(f'{output_path}/labels/{t}')
            game_levels = [f"{t}:{level}\n" for level in game_levels]
            with open(f'{output_path}/splits/{protocal}_test_{s}_fold_{fold}.txt', 'a') as f:
                f.writelines(game_levels)
