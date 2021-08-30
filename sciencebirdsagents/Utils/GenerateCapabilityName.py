import os
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--level_path', type=str, default='fifth_generation')
args = parser.parse_args()
target_level_path = '../sciencebirdslevels/generated_levels/{}'.format(args.level_path)

templates_to_run = []
levels = sorted(os.listdir(target_level_path))
for level in levels:
    capabilities = sorted(os.listdir(os.path.join(target_level_path, level)))
    for capability in capabilities:
        sys.stdout.write("{}_{}\n".format(level, capability))