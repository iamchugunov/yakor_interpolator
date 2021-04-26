import socket
from config import Config
import json
import processing as pr

config = Config()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(config.ADDR)

# 0x150001 - найстройки выстрела
# 0х150002 - массив измерений
# 0х150003 - массив точек траекторий

while True:

    rcv_size = int.from_bytes(client.recv(4), "little")
    rcv_type = int.from_bytes(client.recv(4), "little")

    print("Size {}".format(rcv_size))
    print("Type {:0x}".format(rcv_type))

    data = client.recv(rcv_size) #20000

    last_bytes = rcv_size - len(data)

    while last_bytes > 0:
        data = data + client.recv(last_bytes)
        last_bytes = rcv_size - len(data)

    if rcv_type == 0x150001 or rcv_type == 0x150002:
        data = json.loads((data.decode()))

        if rcv_type == 0x150001:
            pr.process_initial_data(data, config)

        if rcv_type == 0x150002:
            if config.ini_data_flag:
                points = data["meas"]
                print(len(points))
                print(points)
                # pr.process_measurements(data, config)
                # N = len(data['trajPoints'])
                # track = {}
                # points = {}
            else:
                print("idi na xyu")

    # if data.type == "initial data":
    #     #     pr.process_initial_data(data, config)
    #     # elif data.type == "measurements":
    #     #     if pr.process_measurements(data, config):
    #     #         client.send(config.track)
    #     #     else:
    #     #         print("Fail")
    #     # else:
    #     #     print("Type is unknown")

    # data = json.loads((data.decode()))
    # print(type(data))


