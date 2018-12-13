import filter_env
from ddpg import *
import gc
import time, os
from internal_connection import sending, receiving

gc.enable()

EPISODES = 100000
TEST = 10

logfold = time.strftime("%dd%mm_%H%M")
if not os.path.exists(logfold):
    os.makedirs(logfold)

def main():
    #python receiver for receiving returns and states from the Java agent
    print("before receiver")
    receiver = receiving.Receiving()
    reward = receiver.reward 
    print reward
    print("after receiver")    
    
    print("before receiver2")
    receiver = receiving.Receiving()
    reward = receiver.reward 
    print reward
    print("after receiver2")    
    
    sess = tf.InteractiveSession()
    agent = DDPG(sess)
    saver = tf.train.Saver(tf.trainable_variables())

    summaries_var = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var.name)
            summaries_var.append(tf.summary.histogram(var.op.name, var))
    summary_op1 =  tf.summary.merge(summaries_var)
    summary_writer1 = tf.summary.FileWriter(os.path.join(logfold,'summary_vars'), sess.graph)

    print("entering for loop")
    for episode in xrange(EPISODES):
        #1. wait until getting the initial state from the agent
        #TODO making while loop until it meets the condition receiver.isSuspended = False 
        while True:
            time.sleep(0.1)
            print(receiver.isSuspended)
            if not receiver.isSuspended: 
                print("break initial loop")
                break
        #state = env.reset()
        state = receiver.obj_np
        #print "episode:",episode
        
        # Train
        for step in xrange(7):
            action = agent.noise_action(state)
            print(action)
            # 2. passing the action value to the Java agent
            # NOTE I suppose that 'action' as a list e.g. [-35, -35, 100] 
            sender.send(action)

            # 3. wait until getting reward and next_state 
            #TODO making while loop until it meets the condition receiver.isSuspended = False 
            while True:
                time.sleep(0.1)
                if not receiver.isSuspended:
                    print("break loop")
                    break
            # Question : What is for 'done'?
            #next_state,reward,done,_ = env.step(action)
            # 4. receive the reward and next_state
            next_state = receiver.obj_np
            reward = receiver.reward
            done = receiver.done
            print("about to perceive")            
            # 5. assign them using agent.perceive (done in below)
            agent.perceive(state,action,reward,next_state,done)
            print(agent.replay_buffer.buffer[-1])
            state = next_state
            if agent.summary_str2 != None and step % 250 == 0:
                    summary_str1 = sess.run(summary_op1)
                    summary_writer1.add_summary(summary_str1, episode*(3) +step)
                    #a.summary_str2

            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in xrange(TEST):
                while True:
                    if not receiver.isSuspended: 
                        break
                #state = env.reset()
                state = receiver.obj_np
                for j in xrange(7):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    while True:
                        if not receiver.isSuspended:
                            break

                    state = receiver.obj_np
                    reward = receiver.reward
                    done = receiver.done
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward

if __name__ == '__main__':
    main()
