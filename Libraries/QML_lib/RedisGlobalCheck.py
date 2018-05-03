
import RedisSettings as rds
import argparse
import redis 

parser = argparse.ArgumentParser(description='Redis settings.')

parser.add_argument(
  '-rh', '--redis_host_name', 
  help="Redis host.",
  type=str,
  default='localhost'
)

parser.add_argument(
  '-rpn', '--redis_port_number', 
  help="Redis port number.",
  type=int,
  default=6379
)

parser.add_argument(
  '-rqid', '--redis_qmd_id', 
  help="QMD ID.",
  type=str,
  default=0
)

parser.add_argument(
  '-g', '--global_host', 
  help="QMD ID.",
  type=str,
  default=0
)

parser.add_argument(
  '-a', '--action', 
  help="QMD ID.",
  type=str,
)


arguments = parser.parse_args()

global_host = arguments.global_host
host_name = str(arguments.redis_host_name)
port_number = arguments.redis_port_number
qid = arguments.redis_qmd_id
action = arguments.action


redis_conn = redis.Redis(host=global_host, port=port_number)

try:
    redis_conn.keys()
except:
    print("Cannot find Global database on", global_host)


global_keys = list(a.decode() for a in redis_conn.keys())

#print("Global keys:", global_keys)
#print("Action:", action)

if action=='add':
    if str(host_name) in global_keys:
        print(host_name, "in keys of global db already")
        val = int(redis_conn.get(host_name))
        val += int(qid)
        print("SETTING (overwriting)", host_name, "on global server to", val)
        redis_conn.set(host_name, val)
        
    else:
        print("SETTING (initial)", host_name, ":", host_name, "on global server to ", qid)
        print("Redis connection:", redis_conn)
        redis_conn.set(host_name, qid)

    
    
elif action=='remove':
    if str(host_name) in global_keys:
        val = int(redis_conn.get(host_name))
        val -= int(qid)
        redis_conn.set(host_name, val)
        print("SETTING (removing)", host_name, " on global server to", val)
    else:
        print(host_name, " not present on global database.")        

       
elif action=='check-end':
    try:
        val = int(redis_conn.get(host_name))
        if val == 0:
            print("redis-finished")
        else: 
            print("redis-running")
    except:
        print("redis-finished")



elif action=='check':
    if str(host_name) in global_keys:
        if int(redis_conn.get(host_name))== 0:
            print("redis-not-ready")
        else:
            print("redis-ready")
    else:
        print("redis-not-ready")

else:
    print("Action must be either add, remove or check")
