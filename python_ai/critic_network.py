from hp import Hp
import tensorflow as tf 
import numpy as np
import math


class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self,sess,state_dim,action_dim, state_input, target_state_input):
		self.time_step = 0
		self.sess = sess
		# create q network
		self.action_input,\
		self.q_value_output,\
		self.net = self.create_q_network(state_dim,action_dim, state_input)

		# create target q network (the same structure with q network)
		self.target_action_input,\
		self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim, self.net, target_state_input)

		self.create_training_method()

		# initialization 
		self.sess.run(tf.global_variables_initializer())
			
		self.update_target()

	def create_training_method(self):
		# Define training optimizer
		self.y_input = tf.placeholder("float",[None,1])
		weight_decay = tf.add_n([Hp.cL2 * tf.nn.l2_loss(var) for var in self.net])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(Hp.cLEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_q_network(self,state_dim,action_dim,state_input):
		# the layer size could be changed
		layer1_size = Hp.cLAYER1_SIZE
		layer2_size = Hp.cLAYER2_SIZE

		#state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])
		with tf.name_scope('critic'):
			W1 = self.variable([state_dim,layer1_size],state_dim)
			b1 = self.variable([layer1_size],state_dim)
			W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
			W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
			b2 = self.variable([layer2_size],layer1_size+action_dim)
			W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
			b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)

		return action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

	def create_target_q_network(self,state_dim,action_dim,net,target_state_input):
		#state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1-Hp.TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(target_state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])

		return action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,y_batch, state_var,state_batch,action_batch):
		self.time_step += 1
		d1 = {self.y_input:y_batch, self.action_input:action_batch}
		d2 = {i: d for i, d in zip(state_var, state_batch)}
		d1.update(d2)
		self.sess.run(self.optimizer,feed_dict=d1)

	def gradients(self,state_var,state_batch,action_batch):
		d1 = {self.action_input:action_batch}
		d2 = {i: d for i, d in zip(state_var, state_batch)}
		d1.update(d2)
		val=  self.sess.run(self.action_gradients,feed_dict=d1)
		#print("******", val)
		return val[0]

	def target_q(self, state_var,state_batch,action_batch):
		d1 = {self.target_action_input:action_batch}
		d2 = {i: d for i, d in zip(state_var, state_batch)}
		d1.update(d2)
		return self.sess.run(self.target_q_value_output,feed_dict=d1)

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"

	def save_network(self,time_step):
		print 'save critic-network...',time_step
		self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
		