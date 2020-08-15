#!/bin/bash

cd ~/neuroevo/

num_running=0
num_desired=$1

#echo "num_desired"
#echo $num_desired

#echo "running_pids:"
for line in $(cat logs/running_pids.txt)
do
    #echo $line
    is_running=$(ps -eo pid | grep $line | wc -l)
    num_running=$(($num_running + $is_running))
done

#echo "num_running"
#echo $num_running

num_to_create=$(($num_desired - $num_running))

#echo "num_to_create"
#echo $num_to_create

run_scripts/launch_worker_queues.sh $num_to_create
