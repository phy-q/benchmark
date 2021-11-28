#!/bin/bash
mode=$1
if [ $mode = "within_template" ]; then
  templates=($(python Utils/GenerateTemplateName.py))
else
  templates=($(python Utils/GenerateCapabilityName.py))
fi
for val in $templates; do
  echo running $val
  python OpenAI_StableBaseline_Train.py --template $val --mode $mode --game_version Linux --level_path "fifth_generation"
done