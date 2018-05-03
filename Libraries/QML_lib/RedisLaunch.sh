#!/bin/bash

host=$(hostname)

GLOBAL_SERVER=$1

job_id=$PBS_JOBID
job_number="$(cut -d'.' -f1 <<<"$job_id")"


echo "Inside launch redis script; host=$host; Global server: $GLOBAL_SERVER"

if [ "$host" == "IT067176" ]
then
    echo "Brian's laptop identified -  launching redis"
    running_dir=$(pwd)
    lib_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/Libraries/QML_lib"
    script_dir="/home/bf16951/Dropbox/QML_share_stateofart/QMD/ExperimentalSimulations"
    SERVER_HOST='localhost'
    ~/redis-4.0.8/src/redis-server  $lib_dir/RedisConfig.conf & 
        
elif [[ "$host" == "newblue"* ]]
then
    echo "BC frontend identified"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
    module load tools/redis-4.0.8
    module load mvapich/gcc/64/1.2.0-qlc
    echo "launching redis"
    redis-server $lib_dir/RedisConfig.conf --protected-mode no  &
    SERVER_HOST='localhost'


elif [[ "$host" == "node"* ]]
then
    echo "BC backend identified"
    running_dir=$(pwd)
    lib_dir="/panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib"
    script_dir="/panfs/panasas01/phys/bf16951/QMD/ExperimentalSimulations"
	par_dir="/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment"
    module load tools/redis-4.0.8
    module load languages/intel-compiler-16-u2
	SERVER_HOST=$(hostname)
    echo "launching redis: $lib_dir/RedisConfig.conf on $SERVER_HOST"
	cd $lib_dir    
	echo "Running check on redisgGlobal"
	#python3 RedisGlobalCheck.py -rh=$SERVER_HOST -rqid=$QMD_ID -g=$GLOBAL_SERVER -a='check'
	REDIS_URL=redis://$SERVER_HOST:6379
	echo "REDIS_URL is $REDIS_URL"

	redis_run_test=`python3 RedisGlobalCheck.py -rh=$SERVER_HOST -rqid=$QMD_ID -g=$GLOBAL_SERVER -a='check'`
	redis_test=$(echo $redis_run_test)	
	#redis_test=""	
	echo "redis test: $redis_test"

	if [[ "$redis_test" == "redis-ready" ]]
	then
		echo "Redis server already present on $SERVER_HOST at $(date +%H:%M:%S)"
	else 
		echo "Redis server NOT already present on $SERVER_HOST; launching at $(date +%H:%M:%S)"
		rm dump.rdb		
		#redis-cli shutdown		
		redis-server RedisDatabaseConfig.conf --protected-mode no &
		echo "Testing launching an RQ worker from inside RedisLaunch script at $(date +%H:%M:%S). Cmd:"
		echo "rq worker $QMD_ID -u $REDIS_URL > $PBS_O_WORKDIR/logs/worker_$job_number.log 2>&1 &"
		rq worker $QMD_ID -u $REDIS_URL >> $PBS_O_WORKDIR/logs/worker_$job_number.log 2>&1 &	

#		./just_launch_redis.sh & 
		echo "Redis server launched on $SERVER_HOST; flushing databases at $(date +%H:%M:%S)."
    	# redis-cli flushall #TODO do flush on first use of this db
	fi
	
	echo "Time: $(date +%H:%M:%S)"

else
    echo "Neither local machine (Brian's university laptop) or blue crystal identified." 
fi

python3 RedisManageServer.py -rh=$SERVER_HOST -rqid=$QMD_ID -action='add'
python3 RedisGlobalCheck.py -rh=$SERVER_HOST -rqid=$QMD_ID --action='add' -g=$GLOBAL_SERVER

echo "Leaving LaunchRedis script at $(date +%H:%M:%S)."

