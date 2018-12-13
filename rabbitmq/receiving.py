#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='AB2BBQ')
def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)

channel.basic_consume(callback,queue='AB2BBQ',no_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
