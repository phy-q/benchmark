#!/bin/bash
fileid="1OvP-w7NIVYEVmQQw0h-HI172k0Iq7whn"
filename="phyre_style_train_data.tar.bz2"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

rm ./cookie
