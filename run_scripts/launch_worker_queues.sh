#!/bin/bash

num_workers=$1

echo "starting ${num_workers} workers"


cd ~/neuroevo

source venvconda3/bin/activate
for ((i=0; i < $num_workers; i++)) # {0..num_workers}
do
    # echo $i
    max_id=$(cat logs/max_id.txt)
    new_id=$(($max_id + 1))
    nohup rq worker -c settings > logs/${new_id}&
    new_pid=$!
    echo $new_pid >> logs/running_pids.txt
    echo ${new_id} > logs/max_id.txt
done
