#!/usr/bin/env python
import pika
import ast
import numpy as np
import json
     
LEN_REDB = 10
LEN_YELLOWB = 10
LEN_BLUEB = 10
LEN_BLACKB = 10
LEN_WHITEB = 10
LEN_PIG = 15
LEN_ICE = 35
LEN_WOOD = 35
LEN_STONE = 35
LEN_TNT = 5
N_COORD = 5

redB = 0
yelB = 0
bluB = 0
blaB = 0
whiB = 0
pig = 0
ice = 0
wood = 0
stone = 0
tnt = 0

class Receiving:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='AB2BBQWJ')
        self.channel.basic_consume(self.callback,queue='AB2BBQWJ',no_ack=True)
        self.obj_np = 0
        self.done = False
        #for how many birds are finally in the state - SHOULD BE ALWAYS 1
        self.cnt = 0
        self.reward = 0
        self.isSuspended = True
        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()    

    def init_raw_state(self):
        redB = np.zeros((LEN_REDB, N_COORD))
        yelB = np.zeros((LEN_YELLOWB, N_COORD))
        bluB = np.zeros((LEN_BLUEB, N_COORD))
        blaB = np.zeros((LEN_BLACKB, N_COORD))
        whiB = np.zeros((LEN_WHITEB, N_COORD))

        pig = np.zeros((LEN_PIG, N_COORD))

        ice = np.zeros((LEN_ICE, N_COORD))
        wood = np.zeros((LEN_WOOD, N_COORD))
        stone = np.zeros((LEN_STONE, N_COORD))
        tnt = np.zeros((LEN_TNT, N_COORD))
        
        raw_state = np.array([redB, yelB, bluB, blaB, whiB, pig, ice, wood, stone, tnt])
        return raw_state

    
    def callback(self, ch, method, properties, body):
        print(" [x] Received %r" % body)
        if body == "TRUE":
            # for notifying the game is successfully ended
            self.done = True
            print ("GAME ENDS")
        elif body == "FALSE":
            self.done = False
            print ("GAME STILL PLAYING")
        elif len(body) <= 7:
            #for receiving rewards
            self.reward = int(body)
            print ("REWARD = ", self.reward)
            self.connection.close()
        else:
            # for receiving states
            body_dict = ast.literal_eval(body)
            counter = [0] * 10
            raw_state = self.init_raw_state()
            for coord in iter(body_dict['objs']):
                #print(coord)
                sample = np.array([coord['x'], coord['y'], coord['width'], coord['height'], coord['angle']])
                category = coord['type']-3
                if category in [0, 1, 2]:
                    if 325 <= coord['y'] <= 345:
                        #valid region for ensuring that the bird is on the sling
                        self.cnt += 1
                        raw_state[category][counter[category]] = sample 
                        counter[category] += 1
                else:
                    raw_state[category][counter[category]] = sample 
                    counter[category] += 1

            if self.cnt  == 0:
                raw_state[2][0] = [176, 335, 8, 8, 0]
                self.cnt += 1
            for i in range(10):
                ind = max(1, counter[i])
                raw_state[i] = np.delete(raw_state[i], np.s_[ind::],0)
            raw_state = np.delete(raw_state, [3, 4, 9])
            self.obj_np = raw_state
            print ("RECEIVED STATE")
            print ("WE MET BIRD , ", self.cnt)
            self.cnt -= 1
    
#self.channel.basic_consume(callback,queue='AB2BBQ',no_ack=True)
#print(' [*] Waiting for messages. To exit press CTRL+C')
#channel.start_consuming()
