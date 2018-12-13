import filter_env
from ddpg import *
import gc
import time, os
from internal_connection import sending, receiving

gc.enable()

EPISODES = 100000
TEST = 1

data_fname = 'transitions.pickle'

logfold = time.strftime("%dd%mm_%H%M")
chkpt_dir = os.path.join(logfold, 'chkpt')
if not os.path.exists(logfold):
    os.makedirs(logfold)
    os.makedirs(os.path.join(logfold,'chkpt'))

def main():
    #python receiver for receiving returns and states from the Java agent
    sender = sending.Sending()
    
    sess = tf.InteractiveSession()
    agent = DDPG(sess, data_fname)
    saver = tf.train.Saver(tf.trainable_variables())
# saver.restore(sess,tf.train.latest_checkpoint('04d06m_1452/chkpt'))
    summaries_var = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var.name)
            summaries_var.append(tf.summary.histogram(var.op.name, var))
    summary_op1 =  tf.summary.merge(summaries_var)
    summary_writer1 = tf.summary.FileWriter(os.path.join(logfold,'summary_vars'), sess.graph)


    for episode in xrange(EPISODES):
        #1. wait until getting the initial state from the agent
        #TODO making while loop until it meets the condition receiver.isSuspended = False 
        receiver = receiving.Receiving()
        state = receiver.obj_np
        print ("Saved State")
        print ("--------------------------------------------------------------------")
        #state = env.reset()
        #print(state)
        #print "episode:",episode
        
        # Train
        for step in xrange(8):
            action = agent.noise_action(state)
            print("my action ",action)
            # 2. passing the action value to the Java agent
            # NOTE I suppose that 'action' as a list e.g. [-35, -35, 100] 
            sender.send(action)

            # 3. wait until getting reward and next_state 
            #TODO making while loop until it meets the condition receiver.isSuspended = False 
            receiver = receiving.Receiving()
            # Question : What is for 'done'?
            #next_state,reward,done,_ = env.step(action)
            # 4. receive the reward and next_state
            next_state = receiver.obj_np
            reward = receiver.reward
            done = receiver.done            
            print ("state saved: ", next_state)
            print ("reward saved: ", reward) 
            print ("--------------------------------------------------------------------")
            print("about to perceive")            
            # 5. assign them using agent.perceive (done in below)
            agent.perceive(state,action,reward,next_state,done)
            #print(agent.replay_buffer.buffer[-1])
            state = next_state
            if agent.summary_str2 != None and step % 250 == 0:
                    summary_str1 = sess.run(summary_op1)
                    summary_writer1.add_summary(summary_str1, episode*(3) +step)
                    #a.summary_str2
            if done:
                break
        
        agent.replay_buffer.save_tuples(data_fname)
        print('buffer length is %d '% len(agent.replay_buffer.buffer))

        # Testing:
        if episode % 5000 == 0 and episode > 100:
            total_reward = 0
            for i in xrange(TEST):
                receiver = receiving.Receiving()
                state = receiver.obj_np
                for j in xrange(7):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    sender.send(action)

                    receiver = receiving.Receiving()
                    state = receiver.obj_np
                    reward = receiver.reward
                    done = receiver.done
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
        
        if episode % 5 == 0:
            checkpoint_path = os.path.join(chkpt_dir, 'model%d.ckpt'%(episode))
            saver.save(sess, checkpoint_path, global_step=episode)
            agent.replay_buffer.save_tuples("bck_"+data_fname)

if __name__ == '__main__':
    main()
