#!/bin/bash

for pid in $(cat ~/neuroevo/logs/running_pids.txt)
do
    kill $pid
done

rm ~/neuroevo/logs/running_pids.txt
echo "-1" > ~/neuroevo/logs/max_id.txt

num_lines=$(cat /etc/crontab | wc -l)

head -n $(($num_lines - 1)) /etc/crontab | sudo tee /etc/crontab
