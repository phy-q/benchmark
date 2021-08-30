#!/bin/bash
mode=$1
if [ $mode = "within_template" ]; then
  templates=($(python Utils/GenerateTemplateName.py --level_path "fifth_generation"))
else
  templates=($(python Utils/GenerateCapabilityName.py --level_path "fifth_generation"))
fi

for val in ${templates}; do
  echo running $val
  python TrainLearningAgent.py --template $val --mode $mode --game_version Linux --level_path "fifth_generation"
done
