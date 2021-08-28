import os
import sys
target_level_path = '../tasks/generated_tasks/'

templates_to_run = []

levels = sorted(os.listdir(target_level_path))
for level in levels:
    capabilities = sorted(os.listdir(os.path.join(target_level_path, level)))
    for capability in capabilities:
        templates = sorted(os.listdir(os.path.join(target_level_path, level, capability)))
        for template in templates:
            templates_to_run.append("{}_{}_{}".format(level, capability, template))

for template in templates_to_run:
    sys.stdout.write(template+"\n")