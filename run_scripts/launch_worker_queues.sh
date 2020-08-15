#!/bin/bash

num_workers=$1

cd ~/neuroevo
source venvconda3/bin/activate
for i in {0..${num_workers}}
do
    max_id=$(cat logs/max_id.txt)
    new_id=$(($max_id + 1))
    nohup rq worker -c settings > logs/${new_id}&
    echo ${new_id} > logs/max_id.txt
done
