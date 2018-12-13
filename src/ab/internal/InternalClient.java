package ab.internal;

import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;

public class InternalClient {

    private String routing_key;
    private static int counter;
    public InternalClient() {
        routing_key = "AB2BBQWJ";
    }
    
    public void send(String QUEUE_) throws Exception {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();

        String queue_name = routing_key + "_" + Integer.toString(counter++);
        channel.queueDeclare(queue_name, false, false, false, null);
        String message = QUEUE_;
        channel.basicPublish("", routing_key, null, message.getBytes("UTF-8"));
        System.out.println(" [x] Sent '" + message + "'");

        channel.close();
        connection.close();
    }
}
