import socket
from config import Config
import json
import processing as pr

config = Config()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt( socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 )
client.connect(config.ADDR)

# 0x150001 - Настройки выстрела
# 0x150002 - Массив измерений
# 0x150003 - Массив точек траектории.
# 

while True:
	rcv_size = int.from_bytes(client.recv(4), "little")
	rcv_type = int.from_bytes(client.recv(4), "little")
	print("Size {}".format(rcv_size))
	print("Type {:0x}".format(rcv_type))
	data = client.recv(rcv_size)

	if rcv_type == 0x150001 or rcv_type == 0x150002:
		data = json.loads((data.decode()))
		print(type(data))
		print(data)
		if rcv_type == 0x150002:
			N = len(data['trajPoints'])
			track = {}
			points = []
			for i in range(N):
				point={}
				point["t"] = N-i
				point["V"] = 10.0*i
				point["Vx"] = 0.
				point["Vy"] = 0.
				point["Vz"] = 0.
				point["A"] = 0.
				point["Ax"] = 0.
				point["Ay"] = 0.
				point["Az"] = 0.
				point["C"] = 0.
				points.append(point)
			track["points"] = points
			track["valid"] = True

			data2send = json.dumps(track).encode()
			client.sendall(len(data2send).to_bytes(4, "little"))
			client.sendall((0x150003).to_bytes(4, "little"))
			client.sendall(data2send)
		
	# if data.type == "initial data":
	#     pr.process_initial_data(data, config)
	# elif data.type == "measurements":
	#     if pr.process_measurements(data, config):
	#         client.send(config.track)
	#     else:
	#         print("Fail")
	# else:
	#     print("Type is unknown")

	# data = json.loads((data.decode()))
	# print(type(data))


