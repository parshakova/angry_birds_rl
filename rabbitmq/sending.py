import pika

connection=pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='BBQ2AB')
channel.basic_publish(exchange='', routing_key='BBQ2AB', body='Hello World!')
print (" [x] Sent 'Hello World!'")

connection.close()
