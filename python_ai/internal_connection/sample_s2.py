import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()


channel.queue_declare(queue='sample')

channel.basic_publish(exchange='',
                      routing_key='sample',
                      body='20')
print(" [x] Sent 'reward'")
connection.close()
