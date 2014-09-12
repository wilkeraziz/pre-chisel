#-*- coding:utf-8 -*-

import zmq
import json

def client(dict_sents):

	context = zmq.Context()

	to_send = json.dumps({'request':dict_sents}) 

	#  Socket to talk to server
	print("Connecting to the server to get the sentences analyzed...")
	socket = context.socket(zmq.REQ)
	socket.connect("tcp://localhost:3001")

	#  Do 10 requests, waiting each time for a respons
	print("Sending request…")
	print(to_send)
	socket.send(to_send)

	#  Get the reply.
	result = socket.recv()
	to_return = json.dumps({'result':result})
	print("Message received...")

	return to_return

if __name__=="__main__":

	client(json.dumps([{'uuid':'1','sentence':'This is .'},{'uuid':'2','sentence':"She does ."}]))