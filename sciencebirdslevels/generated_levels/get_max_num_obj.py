import os
import json

level_path = 'fifth_generation'
cap_levels = os.listdir(level_path)
level_max_num_obj = {}

for cap_l in cap_levels:
    for scenario in os.listdir(f"{level_path}/{cap_l}"):
        for template in os.listdir(f"{level_path}/{cap_l}/{scenario}"):
            for game_level in os.listdir(f"{level_path}/{cap_l}/{scenario}/{template}"):
                with open(f"{level_path}/{cap_l}/{scenario}/{template}/{game_level}") as f:
                    level_file = f.readlines()
                    start_ind = 0
                    end_ind = 0
                    for ind, line in enumerate(level_file):
                        if '<GameObjects>\n' in line:
                            start_ind = ind
                        elif ' </GameObjects>\n' in line:
                            end_ind = ind

                    if start_ind == 0 or end_ind == 0:
                        raise Exception("no game objects found")

                    num_objs = end_ind - start_ind - 1 + 2

                    if f'{cap_l}_{scenario}_{template}' in level_max_num_obj:
                        if level_max_num_obj[f'{cap_l}_{scenario}_{template}'] < num_objs:
                            level_max_num_obj[f'{cap_l}_{scenario}_{template}'] = num_objs
                    else:
                        level_max_num_obj[f'{cap_l}_{scenario}_{template}'] = num_objs

                    if f'{cap_l}_{scenario}' in level_max_num_obj:
                        if level_max_num_obj[f'{cap_l}_{scenario}'] < num_objs:
                            level_max_num_obj[f'{cap_l}_{scenario}'] = num_objs
                    else:
                        level_max_num_obj[f'{cap_l}_{scenario}'] = num_objs

with open('level_max_num_obj.json', 'w') as f:
    json.dump(level_max_num_obj, f)
print(level_max_num_obj)