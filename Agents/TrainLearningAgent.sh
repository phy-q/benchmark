#!/bin/bash
mode=$1
if [ $mode = "within_capability" ]
then
  templates=$(python Utils/GenerateCapabilityName.py)
elif [ $mode = "within_template" ]
then
  templates=$(python Utils/GenerateTemplateName.py)
fi
for val in $templates; do
    echo running $val
    python TrainLearningAgent.py --template $val --mode $mode
done