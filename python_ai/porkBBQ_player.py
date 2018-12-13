import filter_env
from ddpg_off import *
import gc
import numpy as np
import time, os
from internal_connection import sending, receiving

gc.enable()

EPISODES = 100000
data_fname = 'transitions.pickle'

def normalize(trans):
	
	redb_norm = (np.array([ 45.83699362,  97.64607853,   2.09739844,   2.61098947,   0.        ]), np.array([  7.21435945e+01,   1.51552440e+02,   3.45336333e+00, 4.20994775e+00,   1.00000000e-07]))
	yelb_norm = (np.array([  45.10942659,  139.07308398,    5.73852779,    5.26547572,    0.        ]), np.array([  5.64628099e+01,   1.65027949e+02,   6.90958398e+00,6.26857164e+00,   1.00000000e-07]))
	blub_norm = (np.array([ 110.4310912 ,  212.37711562,    5.39061199,    5.23134092,    0.        ]), np.array([  8.79155053e+01,   1.61727477e+02,   1.17951988e+01, 8.02921180e+00,   1.00000000e-07]))
	pig_norm = (np.array([ 575.54126107,  331.80913597,   15.30710594,   12.99517765,    0.        ]), np.array([  7.32398930e+01,   4.03923792e+01,   1.65815997e+01, 1.38314680e+01,   1.00000000e-07]))
	ice_norm = (np.array([ 537.3447642 ,  280.02948496,   13.12872608,   16.57941049,    0.        ]), np.array([  1.28765355e+02,   6.92104618e+01,   1.22493247e+01, 1.38300645e+01,   1.00000000e-07]))
	wood_norm = (np.array([ 577.61370558,  309.32742095,   16.72795302,   13.01085275,    0.        ]), np.array([  5.81673550e+01,   3.48052319e+01,   1.35784557e+01, 1.15324956e+01,   1.00000000e-07]))
	stone_norm = (np.array([ 554.06971222,  295.78539321,   11.69317174,    9.53460383,    0.        ]), np.array([  1.33437976e+02,   7.28283063e+01,   1.10374181e+01,  7.30172247e+00,   1.00000000e-07]))
	return  (np.divide(np.subtract(trans[0], redb_norm[0]), redb_norm[1])[:,:-1], np.divide(np.subtract(trans[1], yelb_norm[0]), yelb_norm[1])[:,:-1], np.divide(np.subtract(trans[2], blub_norm[0]), blub_norm[1])[:,:-1],np.divide(np.subtract(trans[3], pig_norm[0]), pig_norm[1])[:,:-1], np.divide(np.subtract(trans[4], ice_norm[0]), ice_norm[1])[:,:-1], np.divide(np.subtract(trans[5], wood_norm[0]), wood_norm[1])[:,:-1], np.divide(np.subtract(trans[6], stone_norm[0]), stone_norm[1])[:,:-1])

def main():
    #python receiver for receiving returns and states from the Java agent
    sender = sending.Sending()
    
    sess = tf.InteractiveSession()
    agent = DDPG(sess, data_fname)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, tf.train.latest_checkpoint('off14d06m_2115/chkpt'))

    total_reward = 0
    for i in xrange(EPISODES):
        receiver = receiving.Receiving()
        state = receiver.obj_np
        for j in xrange(7):
            #env.render()
            action = agent.action(normalize(state)) # direct action for test
            sender.send(action)

            receiver = receiving.Receiving()
            state = receiver.obj_np
            reward = receiver.reward
            done = receiver.done

            total_reward += reward
            if done:
                receiver.done = False
                break
    ave_reward = total_reward/TEST
    print 'episode: ',episode,'Evaluation Average Reward:',ave_reward


if __name__ == '__main__':
    main()
