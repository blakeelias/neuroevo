#!/bin/bash

for pid in $(cat ~/neuroevo/logs/running_pids.txt)
do
    kill $pid
done

rm ~/neuroevo/logs/running_pids.txt
echo "-1" > ~/neuroevo/logs/max_id.txt
