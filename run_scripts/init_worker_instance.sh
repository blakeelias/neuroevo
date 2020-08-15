#!/bin/bash

num_workers=$1
master_IP=$2

cd ~/neuroevo
git pull origin master

echo "REDIS_HOST = '${master_IP}'" > settings.py

echo "-1" > logs/max_id.txt
run_scripts/cron_check_pids.sh $num_workers
# run_scripts/launch_worker_queues.sh $num_workers

