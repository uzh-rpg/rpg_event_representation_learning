#!/usr/bin/env python
import numpy as np
import pickle 
import rospkg
import rospy
from rospy.numpy_msg import numpy_msg
import socket 
import sys
import time
from celex5_msgs.msg import event, eventData, eventVector

# Define constants
COMM_PORT = 8484
HEADERSIZE = 10

class CommServer:
    msg = None

    def __init__(self, port, out_loc):
        self.port = int(port)
        self.out_loc = out_loc
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind( ("", self.port))
        except socket.error as emsg:
            rospy.logerr("Socket error: ", emsg)
            sys.exit(1)
        self.events_sub = rospy.Subscriber("/celex_monocular/celex5_event", numpy_msg(eventData), self.eventVector_cb, queue_size=1)

    def shutdown(self):
        self.server.shutdown(socket.SHUT_RDWR)
        self.server.close()

    def listen(self):
        self.server.listen(1)
        while True:
            try:
                client, address = self.server.accept()
            except socket.error as emsg:
                rospy.logerr("Socket error: ", emsg)
                self.server.close()
                sys.exit(1)
            self.clientHandler(client, address)
    
    def clientHandler(self, client, address):
        ip, port = address
        id = str(ip) + ":" + str(port)
        rospy.loginfo("New client: %s" % id)
        start_time = time.time()
        while time.time()-start_time < 10.0:
            # Receive request type
            request = client.recv(1)
            if request == b'\x00':
                # msg = pickle.dumps(self.events)
                # rospy.loginfo("sent msg: %d" % len(msg))
                # msg = bytes("{0:<{1}}".format( len(msg), HEADERSIZE).encode('utf-8')) + msg 
                # client.send(msg)
                npX = np.fromiter(self.msg.x, dtype=np.float32) 
                npY = np.fromiter(self.msg.y, dtype=np.float32)
                npT = np.fromiter(self.msg.timestamp, dtype=np.float32)
                npP = np.zeros_like(npT)
                events = np.stack([npX, npY, npT, npP], axis=-1)
                print(events.shape)
                np.save(out_loc, events)
                client.send(b'\x00')

    def eventVector_cb(self, msg):
        self.msg = msg
        print(len(msg.x))

if __name__ == '__main__':
    rospy.init_node("events2npy")
    rospack = rospkg.RosPack()
    out_loc = rospack.get_path('celex5_monocular') + "/output/eventRecord"

    server = CommServer(COMM_PORT, out_loc)

    rospy.on_shutdown(server.shutdown)
    rospy.loginfo("NODE: event_to_txt starts")
    
    while not rospy.is_shutdown():
        server.listen()
        