import json
import socket
import processing as pr

from config import Config

config = Config()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(config.ADDR)

# 0x150001 - найстройки выстрела
# 0х150002 - массив измерений
# 0х150003 - массив точек траекторий
# 0x150004 - сообщение об ошибке

while True:

    rdata = client.recv(4)
    if len(rdata) < 4:
        break
    rcv_size = int.from_bytes(rdata, "little")

    rdata = client.recv(4)
    if len(rdata) < 4:
        break
    rcv_type = int.from_bytes(rdata, "little")

    print("Size {}".format(rcv_size))
    print("Type {:0x}".format(rcv_type))

    data = client.recv(rcv_size)

    if len(data) == 0:
        break
    last_bytes = rcv_size - len(data)

    while last_bytes > 0:
        rdata = client.recv(last_bytes)
        if len(rdata) == 0:
            break
        data = data + rdata
        last_bytes = rcv_size - len(data)

    if rcv_type == 0x150001 or rcv_type == 0x150002:
        data = json.loads((data.decode()))

        if rcv_type == 0x150001:
            # transfer or inital data
            pr.process_initial_data(data, config)

        if rcv_type == 0x150002:
            # transfer of measurements
            if config.ini_data_flag:
                # the input data correct

                pr.process_measurements(data, config, client)
                if config.ini_meas_flag:

                    if config.data_points:
                        data2send = json.dumps(config.track).encode()
                        client.sendall(len(data2send).to_bytes(4, "little"))
                        client.sendall((0x150003).to_bytes(4, "little"))
                        client.sendall(data2send)
                    else:
                        data2send = json.dumps(config.track).encode()
                        client.sendall(len(data2send).to_bytes(4, "little"))
                        client.sendall((0x150004).to_bytes(4, "little"))
                        client.sendall(data2send)
                        print("Error")
                else:
                    data2send = json.dumps(config.track).encode()
                    client.sendall(len(data2send).to_bytes(4, "little"))
                    client.sendall((0x150004).to_bytes(4, "little"))
                    client.sendall(data2send)
                    print("Error")
            else:
                data2send = json.dumps(config.track).encode()
                client.sendall(len(data2send).to_bytes(4, "little"))
                client.sendall((0x150004).to_bytes(4, "little"))
                client.sendall(data2send)
                print("Error")
