package ab.internal;

import com.rabbitmq.client.*;
import java.io.IOException;

public class InternalServer {

    private String routing_key;
    private ConnectionFactory factory;
    private Channel channel;
    private Connection connection;
    public String output ="INIT";
    public Thread sync_thread;
    public boolean suspended = false;

    public InternalServer() throws Exception{
        routing_key = "BBQ2ABWJ";
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        connection = factory.newConnection();
        Channel channel = connection.createChannel();
        String message;
        channel.queueDeclare(routing_key, false, false, false, null);
        
        System.out.println(" [*] RECEIVING CHANNEL IS OPENED ON AB");
        suspended = true;
        Consumer consumer = new DefaultConsumer(channel) {
            @Override
                public void handleDelivery(String consumerTag, Envelope envelope, AMQP.BasicProperties properties, byte[] body)
                throws IOException {
                    String message = new String(body, "UTF-8");
                    System.out.println(" [x] Received '" + message + "'");
                    output = message;
                    suspended = false;
                }
        };
        channel.basicConsume(routing_key, true, consumer);
    } 
    
    public void closeConnection() throws Exception {
        channel.close();
        connection.close();
    }
}
