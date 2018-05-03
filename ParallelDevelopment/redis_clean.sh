#!/bin/bash

ssh node33-010

module load tools/redis-4.0.8
redis-server &
echo "before flushall"
redis-cli ping
redis-cli flushall
redis-cli shutdown

echo "After shutdown"
redis-cli ping

exit
