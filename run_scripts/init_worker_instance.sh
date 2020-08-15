#!/bin/bash

num_workers=$1
master_IP=$2

cd ~/neuroevo
git pull origin master

echo "REDIS_HOST = '${master_IP}'" > settings.py
run_scripts/launch_worker_queues.sh $num_workers

