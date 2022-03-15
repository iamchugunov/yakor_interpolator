import json
import socket
import launch_processing as pr

from config import Config

config = Config()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(config.ADDR)

# message code
# 0x150001 - initial file firing settings
# 0х150002 - projectile flight array
# 0х150003 - projectile flight array final trajectory
# 0x150004 - error message

while True:

    r_data = client.recv(4)
    if len(r_data) < 4:
        break
    rcv_size = int.from_bytes(r_data, "little")

    r_data = client.recv(4)
    if len(r_data) < 4:
        break
    rcv_type = int.from_bytes(r_data, "little")

    print("Size {}".format(rcv_size))
    print("Type {:0x}".format(rcv_type))

    data = client.recv(rcv_size)

    if len(data) == 0:
        break
    last_bytes = rcv_size - len(data)

    while last_bytes > 0:
        r_data = client.recv(last_bytes)
        if len(r_data) == 0:
            break
        data = data + r_data
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

                pr.process_measurements(data, config)
                if config.ini_meas_flag:

                    if config.data_points:
                        data_2_send = json.dumps(config.track).encode()
                        client.sendall(len(data_2_send).to_bytes(4, "little"))
                        client.sendall((0x150003).to_bytes(4, "little"))
                        client.sendall(data_2_send)

                    else:
                        data_2_send = json.dumps(config.track).encode()
                        client.sendall(len(data_2_send).to_bytes(4, "little"))
                        client.sendall((0x150004).to_bytes(4, "little"))
                        client.sendall(data_2_send)
                        print("Error")
                else:
                    data_2_send = json.dumps(config.track).encode()
                    client.sendall(len(data_2_send).to_bytes(4, "little"))
                    client.sendall((0x150004).to_bytes(4, "little"))
                    client.sendall(data_2_send)
                    print("Error")
            else:
                data_2_send = json.dumps(config.track).encode()
                client.sendall(len(data_2_send).to_bytes(4, "little"))
                client.sendall((0x150004).to_bytes(4, "little"))
                client.sendall(data_2_send)
                print("Error")
