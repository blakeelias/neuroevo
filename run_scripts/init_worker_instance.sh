#!/bin/bash

num_workers=$1
master_IP=$2

cd ~/neuroevo
git pull origin master

echo "REDIS_HOST = '${master_IP}'" > settings.py

echo "-1" > logs/max_id.txt
CMD="/home/ubuntu/neuroevo/run_scripts/cron_check_pids.sh $num_workers"
$CMD

# schedule CMD
line="* * * * * ubuntu $CMD > /home/ubuntu/neuroevo/logs/init_log.log; echo hi-from-cron > /home/ubuntu/neuroevo/test.txt"
echo " " | sudo tee -a /etc/crontab
echo "$line" | sudo tee -a /etc/crontab

