import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ab.vision.ABType;
import ab.vision.ABObject;
import ab.vision.Vision;
import ab.vision.real.shape.Body;
import ab.vision.real.shape.Circle;

public class ABObjectSend {

    private final static String QUEUE_NAME = "hello";

    public static void main(String[] argv) throws Exception {

        // Making a list of ABObject with randomized inputs
        // a list includes 3 different ABobjects
        // 1. a red bird with x,y, height, width 
        // 2. pig, type of circle
        // 3. stone, type stone
        /*
        String Json_objList = "{objs:[";
        List<ABObject> obj_list = new ArrayList();
        ABObject redbird = new ABObject(new Rectangle(1, 2, 3, 4), ABType.RedBird);
        ABObject pig = new ABObject(new Rectangle(5, 6, 7, 8), ABType.Pig);
        ABObject stone = new ABObject(new Rectangle(9, 10, 11, 12), ABType.Stone);

        obj_list.add(redbird);
        obj_list.add(pig);
        obj_list.add(stone);
       
        for (int i = 0; i < obj_list.size(); i++) {
            Json_objList.concat(obj_list.get(i).toJson());
            if(i < obj_list.size()-1)
                Json_objList.concat(",");
        }
        Json_objList.concat("]");
        */

        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();

        channel.queueDeclare(QUEUE_NAME, false, false, false, null);
        String message = "HELLO_WORLD";
        channel.basicPublish("", QUEUE_NAME, null, message.getBytes("UTF-8"));
        System.out.println(" [x] Sent '" + message + "'");

        channel.close();
        connection.close();
    }

}
