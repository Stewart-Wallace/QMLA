#!/bin/bash

echo "launching redis on $(hostname)" 

module load tools/redis-4.0.8
redis-cli shutdown
redis-server RedisDatabaseConfig.conf --protected-mode no   & 

