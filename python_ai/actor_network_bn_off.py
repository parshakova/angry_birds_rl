import tensorflow as tf 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math
from hp import Hp


class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim, mystate_input, state_input, target_state_input, state_net):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.action_output,self.net,self.is_training = self.create_network(state_dim,action_dim, state_input)
		self.state_input = mystate_input

		self.state_net = state_net

		# create target actor network
		self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.net, target_state_input)

		# define training rules
		self.create_training_method()

		self.sess.run(tf.global_variables_initializer())

		self.summary_op2 =  tf.summary.merge(self.summaries)
    	

		self.update_target()
		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		params = self.net + self.state_net

		self.parameters_gradients = tf.gradients(self.action_output,params,-self.q_gradient_input)
		self.summaries = []
		for grad, var in zip(self.parameters_gradients,params):     
			self.summaries.append(tf.summary.histogram(var.op.name + '/grad', grad))

		self.optimizer = tf.train.AdamOptimizer(Hp.aLEARNING_RATE).apply_gradients(zip(self.parameters_gradients,params))

	def create_network(self,state_dim,action_dim, state_input):
		layer1_size = Hp.aLAYER1_SIZE
		layer2_size = Hp.aLAYER2_SIZE

		#state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)
		with tf.name_scope('actor'):
			W1 = self.variable([state_dim,layer1_size],state_dim)
			b1 = self.variable([layer1_size],state_dim)
			W2 = self.variable([layer1_size,layer2_size],layer1_size)
			b2 = self.variable([layer2_size],layer1_size)
			W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
			b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		layer0_bn = self.batch_norm_layer(state_input,training_phase=is_training,scope_bn='batch_norm_0',activation=tf.identity)
		layer1 = tf.matmul(layer0_bn,W1) + b1
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,W2) + b2
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='batch_norm_2',activation=tf.nn.relu)

		action_output = tf.sigmoid(tf.matmul(layer2_bn,W3) + b3)

		return action_output,[W1,b1,W2,b2,W3,b3],is_training

	def create_target_network(self,state_dim,action_dim,net, target_state_input):
		#state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)
		ema = tf.train.ExponentialMovingAverage(decay=1-Hp.TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer0_bn = self.batch_norm_layer(target_state_input,training_phase=is_training,scope_bn='target_batch_norm_0',activation=tf.identity)

		layer1 = tf.matmul(layer0_bn,target_net[0]) + target_net[1]
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='target_batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,target_net[2]) + target_net[3]
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='target_batch_norm_2',activation=tf.nn.relu)
		# output dx, dy, bird-tripple
		action_output = tf.sigmoid(tf.matmul(layer2_bn,target_net[4]) + target_net[5])

		return  action_output,target_update,is_training

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_var,state_batch):
		d1 = {self.q_gradient_input:q_gradient_batch,self.is_training: True}
		d2 = {i: d for i, d in zip(state_var, state_batch)}
		d1.update(d2)
		_, summary_str2 = self.sess.run([self.optimizer,self.summary_op2],feed_dict=d1)
		return summary_str2

	def actions(self,state_var,state_batch):
		d1 = {self.is_training: True}
		d2 = {i: d for i, d in zip(state_var, state_batch)}
		d1.update(d2)
		return self.sess.run(self.action_output,feed_dict=d1)

	def action(self,state):
		d1 = {self.is_training: False}
		d2 = {i: d for i, d in zip(self.state_input, state)}
		d1.update(d2)
		val = self.sess.run(self.action_output,feed_dict=d1)
		#print(val)
		return val[0]


	def target_actions(self,state_var, state_batch):
		d1 = {self.target_is_training: True}
		d2 = {i: d for i, d in zip(state_var, state_batch)}
		d1.update(d2)
		return self.sess.run(self.target_action_output,feed_dict=d1)

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


	def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
		return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
		updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
		updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

		
