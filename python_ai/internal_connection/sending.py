#!/usr/bin/env python
import pika

class Sending:
    def __init__(self):
        self.connection=pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='BBQ2AB')
    
    def list2str(action_list):
        str_list = ','.join(str(e) for e in action_list)
        return str_list

    def send(self, action_list):
        message = ",".join(map(str,action_list))
        # format of message should be "-30,30,1000"
        self.channel.basic_publish(exchange='', routing_key='BBQ2ABWJ', body=message)
        print (" [x] Sent ", message)

    def close(self):
        self.connection.close()
