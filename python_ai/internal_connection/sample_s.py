import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()


channel.queue_declare(queue='AB2BBQWJ')

channel.basic_publish(exchange='',
                      routing_key='AB2BBQWJ',
                      body='20')
print(" [x] Sent 'reward'")
connection.close()
