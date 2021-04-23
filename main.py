import socket
from config import Config
import json
import processing as pr

config = Config()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(config.ADDR)

while True:
    data = client.recv(1000)
    if data.type == "initial data":
        pr.process_initial_data(data, config)
    elif data.type == "measurements":
        if pr.process_measurements(data, config):
            client.send(config.track)
        else:
            print("Fail")
    else:
        print("Type is unknown")

    # data = json.loads((data.decode()))
    # print(type(data))


