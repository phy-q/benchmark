#!/bin/bash
#conda activate pytorch
model=$1
splits=$(ls ../data/splits)

for file in $splits
do 
	protocal=$(echo $file | awk 'BEGIN {FS = "_"} {print $1}')
	if [ $protocal = "template" ]
	then
		t1=$(echo $file | awk 'BEGIN {FS = "_"} {print $3}')
		t2=$(echo $file | awk 'BEGIN {FS = "_"} {print $4}')
		t3=$(echo $file | awk 'BEGIN {FS = "_"} {print $5}')
		t=$t1"_"$t2"_"$t3
		fold=$(echo $file | awk 'BEGIN {FS = "_"} {print $7}')
	else
                t1=$(echo $file | awk 'BEGIN {FS = "_"} {print $3}')
                t2=$(echo $file | awk 'BEGIN {FS = "_"} {print $4}')
                #t3=$(echo $file | awk 'BEGIN {FS = "_"} {print $5}')
                t=$t1"_"$t2
		fold=$(echo $file | awk 'BEGIN {FS = "_"} {print $6}')
	fi
	echo $protocal $t ${fold:0:1}
	python train.py --cfg configs/rpcin_within_pred.yaml --template $t --protocal $protocal --fold ${fold:0:1} --model $model
done
