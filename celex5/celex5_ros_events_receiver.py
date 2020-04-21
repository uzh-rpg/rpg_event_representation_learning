#! /usr/bin/env python3
import socket 

# Define constants
COMM_IP = 'localhost'
COMM_PORT = 8484 

if __name__ == '__main__': 
	client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	client.connect((COMM_IP,COMM_PORT))
	while True:	
		client.send(b'\x00') 
		print(client.recv(1))
	client.close()

