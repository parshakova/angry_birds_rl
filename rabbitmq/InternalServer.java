import com.rabbitmq.client.*;

import java.io.IOException;

public class InternalServer {

    private String routing_key;
    private ConnectionFactory factory;
    private Channel channel;

    public InternalServer() {
        routing_key = "BBQ2AB"
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();

        channel.queueDeclare(QUEUE_NAME, false, false, false, null);
        
        System.out.println(" [*] RECEIVING CHANNEL IS OPENED ON AB");
    }

    public void recv(String QUEUE_) throws Exception {
        Consumer consumer = new DefaultConsumer(channel) {
            @Override
                public void handleDelivery(String consumerTag, Envelope envelope, AMQP.BasicProperties properties, byte[] body)
                throws IOException {
                    String message = new String(body, "UTF-8");
                    System.out.println(" [x] Received '" + message + "'");
                }
        };
        channel.basicConsume(QUEUE_NAME, true, consumer);
 
    }
    
    public void closeConnection() throws Exception {
        channel.close();
        connection.close();
    }
}
