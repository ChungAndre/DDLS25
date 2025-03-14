#!/bin/bash
cd "$(dirname "$0")" || return
START_TIME=$SECONDS

for ((i=0; i<6; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u "DDPD.py" $i>"out$i.txt") &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"
