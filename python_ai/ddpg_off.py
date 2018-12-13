# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network_off import CriticNetwork 
from actor_network_bn_off import ActorNetwork
from replay_buffer import ReplayBuffer
from hp import Hp
from state_encoder import StateEnc


class DDPG:
    """docstring for DDPG"""
    def __init__(self, sess, data_fname,replay=False):
        self.name = 'DDPG' 
        # Randomly initialize actor network and critic network
        # with both their target networks

        self.name = 'DDPG' # name for uploading results
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = Hp.state_dim
        self.action_dim = Hp.action_dim
        print(self.state_dim, self.action_dim)

        self.sess = sess

        self.state_input = [tf.placeholder(tf.float32, shape=(None, None, Hp.n_coord)) for _ in xrange(Hp.categories)]
        #tf.placeholder("float",[None,self.state_dim])
        self.target_state_input = [tf.placeholder(tf.float32, shape=(None, None, Hp.n_coord)) for _ in xrange(Hp.categories)]
        #tf.placeholder("float",[None,self.state_dim])
        self.state_network = StateEnc(self.sess, self.state_input, self.target_state_input)
        state_batch = self.state_network.encoding
        next_state_batch = self.state_network.target_encoding

        weights, biases, w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2 = self.state_network.get_parameters()

        state_network_params = weights + biases+ [w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2]

        self.actor_network = ActorNetwork(self.sess,Hp.n_hidden,self.action_dim, self.state_input, state_batch, next_state_batch, state_network_params)
        self.critic_network = CriticNetwork(self.sess,Hp.n_hidden,self.action_dim, state_batch, next_state_batch)
        
        # initialize replay buffer
        if replay:
        	self.replay_buffer = ReplayBuffer(Hp.REPLAY_BUFFER_SIZE, data_fname)
        self.summary_str2 = None

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)


    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatches = self.replay_buffer.get_batch(Hp.batch_size*Hp.N_TRAIN)
        print("######## TRAINING #########")
        for k in range(Hp.N_TRAIN):
            minibatch = minibatches[k*Hp.batch_size:(k+1)*Hp.batch_size]
            state_batch_r = np.asarray([data[0] for data in minibatch])
            state_batch = []
            for j in range(Hp.categories):
                new_cat = np.stack(state_batch_r[:,j],axis=0)
                state_batch.append(new_cat)
            #state_batch = [np.expand_dims(state_batch, axis=1)]
            action_batch = np.asarray([data[1] for data in minibatch])
            reward_batch = np.asarray([data[2] for data in minibatch])
            next_state_batch_r = np.asarray([data[3] for data in minibatch])
            next_state_batch = []
            for j in range(Hp.categories):
                new_cat = np.stack(next_state_batch_r[:,j],axis=0)
                next_state_batch.append(new_cat)
            #next_state_batch = [np.expand_dims(next_state_batch, axis=1)]
            done_batch = np.asarray([data[4] for data in minibatch])

            # for action_dim = 1
            action_batch = np.resize(action_batch,[Hp.batch_size,self.action_dim])
            
            next_action_batch = self.actor_network.target_actions(self.target_state_input, next_state_batch)
            q_value_batch = self.critic_network.target_q(self.target_state_input, next_state_batch,next_action_batch)
            y_batch = []  

            for i in range(len(minibatch)): 
                if done_batch[i]:
                    y_batch.append(reward_batch[i])
                else :
                    y_batch.append(reward_batch[i] + Hp.GAMMA * q_value_batch[i])

            y_batch = np.resize(y_batch,[Hp.batch_size,1])

            # Update critic by minimizing the loss L
            self.critic_network.train(y_batch,self.state_input,state_batch,action_batch)

            # Update the actor policy using the sampled gradient:
            action_batch_for_gradients = self.actor_network.actions(self.state_input,state_batch)
            q_gradient_batch = self.critic_network.gradients(self.state_input,state_batch,action_batch_for_gradients)

            self.summary_str2 =  self.actor_network.train(q_gradient_batch,self.state_input,state_batch)
            
            # Update the target networks
            self.actor_network.update_target()
            self.critic_network.update_target()
            self.state_network.update_target()

    def train_off(self, minibatch):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        state_batch_r = np.asarray([data[0] for data in minibatch])
        state_batch = []
        for j in range(Hp.categories):
            new_cat = np.stack(state_batch_r[:,j],axis=0)
            state_batch.append(new_cat)
        #state_batch = [np.expand_dims(state_batch, axis=1)]
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch_r = np.asarray([data[3] for data in minibatch])
        next_state_batch = []
        for j in range(Hp.categories):
            new_cat = np.stack(next_state_batch_r[:,j],axis=0)
            next_state_batch.append(new_cat)
        #next_state_batch = [np.expand_dims(next_state_batch, axis=1)]
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[Hp.batch_size,self.action_dim])
        
        next_action_batch = self.actor_network.target_actions(self.target_state_input, next_state_batch)
        q_value_batch = self.critic_network.target_q(self.target_state_input, next_state_batch,next_action_batch)
        y_batch = []  

        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + Hp.GAMMA * q_value_batch[i])

        y_batch = np.resize(y_batch,[Hp.batch_size,1])

        # Update critic by minimizing the loss L
        cost, self.summary_str2 = self.critic_network.train_off(y_batch,self.state_input,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(self.state_input,state_batch)
        q_gradient_batch = self.critic_network.gradients(self.state_input,state_batch,action_batch_for_gradients)

        summary_str3 =  self.actor_network.train(q_gradient_batch,self.state_input,state_batch)
        
        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()
        self.state_network.update_target()
        return cost



    def action(self,state):
        state = [np.expand_dims(el, axis=0) for el in state]
        action = self.actor_network.action(state)
        return np.multiply(action, np.array([-35.0,35.0,2000.0]))







