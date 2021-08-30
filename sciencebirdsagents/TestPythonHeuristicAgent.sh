#!/bin/bash
agent="$1"
templates=($(python Utils/GenerateTemplateName.py))
for val in $templates; do
    echo running $val
    python MultiAgentTestOnly.py --template $val --agent $agent
done