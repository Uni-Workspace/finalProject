import socket

ip = "172.18.226.117"
port = 8888

server = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
server.bind((ip, port))

data = server.recv(16).decode()
print(data
)