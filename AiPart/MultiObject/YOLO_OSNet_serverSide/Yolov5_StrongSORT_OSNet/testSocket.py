import socket

host = "172.18.227.249"
port = 5000  # initiate port no above 1024

server_socket = socket.socket()  # get instance
# look closely. The bind() function takes tuple as argument
server_socket.bind((host, port))  # bind host address and port together

# configure how many client the server can listen simultaneously
server_socket.listen(2)
conn, address = server_socket.accept()  # accept new connection
print("Connection from: " + str(address))


while True:
    # receive data stream. it won't accept data packet greater than 1024 bytes
    data = conn.recv(16).decode()
    if not data:
        # if data is not received break
        break
    print("from connected user: " + str(data))
    data = data.strip('][').split(', ')
    
    print(data[0], data[1], data[2])
    print(type(data[0]), type(data[1]), type(data[2]))