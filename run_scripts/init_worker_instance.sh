#!/bin/bash

num_workers=$1
master_IP=$2

cd ~/neuroevo
git pull origin master

echo "REDIS_HOST = '${master_IP}'" > settings.py

echo "-1" > logs/max_id.txt
CMD="run_scripts/cron_check_pids.sh $num_workers"
$(CMD)

# schedule CMD
line="* * * * * ubuntu $CMD""
sudo echo $line >> /etc/crontab


