import tensorflow as tf
from LSTMCell import BasicLSTMCell2
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import array_ops, tensor_array_ops, io_ops
from tensorflow.python.framework import tensor_shape
from hp import Hp


class StateEnc:

	def __init__(self, sess, state_input, target_state_input):

		self.sess= sess
		self.in_state = state_input
		self.target_in_state = target_state_input
		self._encode(state_input)
		self._target_encode(target_state_input)

		self.sess.run(tf.global_variables_initializer())


	def _encode(self, in_state):
		"""
		categories = ['rbird', 'bbird', 'pig', 'wood', 'ice', 'stone']

		in_state is an object, received from java pipe, with following structure of each field:
			- category of an object, e.g. 'ice block','pig' : batch of list of tuples with position information if such object is present (x1,y1,x2,y2,alpha)

		E.G. meta data in_state = {'pig': (batch, seqlen, n_coord) for number of pigs = seqlen 
							    'bird': ...}

		"""
		#self.in_state = [tf.placeholder(tf.float32, shape=(Hp.batch_size, None, Hp.state_dim)) for _ in xrange(Hp.categories)]
		lstm1 = BasicLSTMCell2(Hp.input_proj, Hp.n_hidden, state_is_tuple=True)
		lstm2 = BasicLSTMCell2(Hp.n_hidden, Hp.n_hidden, state_is_tuple=True)

		with tf.variable_scope('lstm'):
			stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

			att_stacked_lstm = BasicLSTMCell2(Hp.n_hidden,Hp.n_hidden, state_is_tuple=True)
			#att_stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(5)], state_is_tuple=True)

		self.weights, self.biases = [], []
		
		with tf.name_scope('encoder_proj'):
			for j in range(Hp.categories):
				self.weights += [tf.Variable(tf.truncated_normal([Hp.n_coord, Hp.input_proj]))]
				self.biases += [tf.Variable(tf.truncated_normal([Hp.input_proj]))]

		initial_state = stacked_lstm.zero_state(Hp.batch_size, tf.float32)

		outputs = []

		for cat in range(Hp.categories):
			# loop along all sequence of items of given category
			rnn_inputs = tf.einsum("ijk,ka->ija", in_state[cat], self.weights[cat]) + self.biases[cat]
			with tf.variable_scope('lstm', reuse= cat>0):
				output, state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=initial_state)

			print(rnn_inputs.get_shape().as_list(), output.get_shape().as_list(),state[Hp.n_layers-1][0].get_shape().as_list(), state[Hp.n_layers-1][1].get_shape().as_list())
			val = tf.transpose(output, [1, 0, 2])
			input_ta = tensor_array_ops.TensorArray(tf.float32, size=tf.shape(val)[0], dynamic_size=True,clear_after_read = True)
			valo = input_ta.unstack(val) #each element is (batch, sentlen)
			last = valo.read(tf.shape(val)[0]-1) 
			outputs.append(tf.expand_dims(last,1)) # last layer lstm output of dynamic_rnn

		# attention mechanism over different categories of objects
		encoded = tf.concat(outputs, 1) 
		state = att_stacked_lstm.zero_state(Hp.batch_size, tf.float32) 
		print(encoded.get_shape().as_list(), state[0].get_shape().as_list())
		for cat in range(Hp.categories):
			a = tf.nn.softmax(tf.einsum("ijk,ik->ij", encoded, state[0])) 
			a_sum = tf.reduce_sum(tf.expand_dims(a,-1)*encoded, axis=1) 
			with tf.variable_scope('lstm', reuse= cat>0):
				output, state = att_stacked_lstm(a_sum, state)

		self.encoding = output

	def _target_encode(self, target_in_state):
		"""
		categories = ['rbird', 'bbird', 'pig', 'wood', 'ice', 'stone']

		in_state is an object, received from java pipe, with following structure of each field:
			- category of an object, e.g. 'ice block','pig' : batch of list of tuples with position information if such object is present (x1,y1,x2,y2,alpha)

		E.G. meta data in_state = {'pig': (batch, seqlen, n_coord) for number of pigs = seqlen 
							    'bird': ...}

		"""
		#self.target_in_state = [tf.placeholder(tf.float32, shape=(Hp.batch_size, None, Hp.state_dim)) for _ in xrange(Hp.categories)]

		weights, biases, w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2 = self.get_parameters()

		net = weights + biases+ [w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2]

		ema = tf.train.ExponentialMovingAverage(decay=1-Hp.TAU)
		self.target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		outputs = []

		# index w: 0 -- Hp.categories-1
		# b: Hp.categories  -- 2*Hp.categories - 1


		lstm1 = BasicLSTMCell2(Hp.input_proj, Hp.n_hidden, weights = target_net[2*Hp.categories:2*Hp.categories+3], state_is_tuple=True)
		lstm2 = BasicLSTMCell2(Hp.n_hidden, Hp.n_hidden, weights = target_net[2*Hp.categories+3:2*Hp.categories+6], state_is_tuple=True)

		with tf.variable_scope('lstm'):
			stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

			att_stacked_lstm = BasicLSTMCell2(Hp.n_hidden,Hp.n_hidden, weights = target_net[2*Hp.categories+6:], state_is_tuple=True)

		initial_state = stacked_lstm.zero_state(Hp.batch_size, tf.float32)

		for cat in range(Hp.categories):
			# loop along all sequence of items of given category
			rnn_inputs = tf.einsum("ijk,ka->ija", target_in_state[cat], target_net[cat]) + target_net[Hp.categories + cat]
			with tf.variable_scope('lstm', reuse= cat>0):
				output, state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=initial_state)

			print(rnn_inputs.get_shape().as_list(), output.get_shape().as_list(),state[Hp.n_layers-1][0].get_shape().as_list(), state[Hp.n_layers-1][1].get_shape().as_list())
			val = tf.transpose(output, [1, 0, 2])
			input_ta = tensor_array_ops.TensorArray(tf.float32, size=tf.shape(val)[0], dynamic_size=True,clear_after_read = True)
			valo = input_ta.unstack(val) #each element is (batch, sentlen)
			last = valo.read(tf.shape(val)[0]-1) 
			outputs.append(tf.expand_dims(last,1)) # last layer lstm output of dynamic_rnn

		# attention mechanism over different categories of objects
		encoded = tf.concat(outputs, 1) 
		state = att_stacked_lstm.zero_state(Hp.batch_size, tf.float32) 
		print(encoded.get_shape().as_list(), state[0].get_shape().as_list())
		for cat in range(Hp.categories):
			a = tf.nn.softmax(tf.einsum("ijk,ik->ij", encoded, state[0])) 
			a_sum = tf.reduce_sum(tf.expand_dims(a,-1)*encoded, axis=1) 
			with tf.variable_scope('lstm', reuse= cat>0):
				output, state = att_stacked_lstm(a_sum, state)

		self.target_encoding = output


	def encode(self, in_state):
		return self.sess.run(self.encoding,feed_dict={i: d for i, d in zip(self.in_state, in_state)}) 

	def target_encode(self, in_state):
		return self.sess.run(self.target_encoding,feed_dict={i: d for i, d in zip(self.target_in_state, in_state)}) 

	def update_target(self):
		self.sess.run(self.target_update)

	def get_parameters(self):

		with tf.variable_scope('lstm/rnn/multi_rnn_cell/cell_0/basic_lstm_cell', reuse =True):
			w_i2h0 = tf.get_variable('w_i2h')
			w_h2h0 = tf.get_variable('w_h2h')
			w_b0 = tf.get_variable('w_b')

		with tf.variable_scope('lstm/rnn/multi_rnn_cell/cell_1/basic_lstm_cell', reuse =True):
			w_i2h1 = tf.get_variable('w_i2h')
			w_h2h1 = tf.get_variable('w_h2h')
			w_b1 = tf.get_variable('w_b')

		with tf.variable_scope('lstm/basic_lstm_cell', reuse=True):
			w_i2h2 = tf.get_variable('w_i2h')
			w_h2h2 = tf.get_variable('w_h2h')
			w_b2 = tf.get_variable('w_b')


		return [self.weights, self.biases, w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2]



