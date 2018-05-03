import redis
import socket

hostname = socket.gethostname()


global_redis = redis.Redis(host=hostname, port=6379)

global_keys = global_redis.keys()

for k in global_keys:
	print("Node", k.decode(), ":", global_redis.get(k).decode())


