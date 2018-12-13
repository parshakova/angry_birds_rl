"""

	3 layered LSTM to encode current state of AIbird agent


"""


import tensorflow as tf
from LSTMCell import BasicLSTMCell2
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import array_ops, tensor_array_ops, io_ops
from tensorflow.python.framework import tensor_shape

n_hidden = 3
input_size = 2
n_layers = 2
n_coord = 5 		#(x1,y1,x2,y2,alpha)
batch_size = 2
categories = 6



def encode(in_state):
	"""
	categories = ['rbird', 'bbird', 'pig', 'wood', 'ice', 'stone']

	in_state is an object, received from java pipe, with following structure of each field:
		- category of an object, e.g. 'ice block','pig' : batch of list of tuples with position information if such object is present (x1,y1,x2,y2,alpha)

	E.G. meta data in_state = {'pig': (batch, seqlen, n_coord) for number of pigs = seqlen 
						    'bird': ...}

	"""

	

		#lstm = lambda: BasicLSTMCell2(input_size, n_hidden, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
	with tf.variable_scope('weights'):
              w_i2h = tf.get_variable('w_i2h', (input_size, 4*n_hidden), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
              w_h2h = tf.get_variable('w_h2h', (n_hidden, 4*n_hidden), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
              w_b = tf.get_variable('w_b', (1, 4*n_hidden), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
	lstm1 = BasicLSTMCell2(input_size, n_hidden, weights=[w_i2h,w_h2h,w_b], state_is_tuple=True)
	lstm2 = BasicLSTMCell2(n_hidden, n_hidden, state_is_tuple=True)
	with tf.variable_scope('lstm'):
		stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

		att_stacked_lstm = BasicLSTMCell2(n_hidden,n_hidden, state_is_tuple=True)
		#att_stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(5)], state_is_tuple=True)



	initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

	weights, biases = [], []

	for j in range(categories):
		weights += [tf.Variable(tf.truncated_normal([n_coord, input_size]))]
		biases += [tf.Variable(tf.truncated_normal([input_size]))]

	outputs = []

	for cat in range(categories):
		# loop along all sequence of items of given category
		rnn_inputs = tf.einsum("ijk,ka->ija", in_state[cat], weights[cat]) + biases[cat]
		with tf.variable_scope('lstm', reuse= cat>0):
			output, state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=initial_state)

		print(rnn_inputs.get_shape().as_list(), output.get_shape().as_list(),state[n_layers-1][0].get_shape().as_list(), state[n_layers-1][1].get_shape().as_list())
		val = tf.transpose(output, [1, 0, 2])
		input_ta = tensor_array_ops.TensorArray(tf.float32, size=tf.shape(val)[0], dynamic_size=True,clear_after_read = True)
		valo = input_ta.unstack(val) #each element is (batch, sentlen)
		last = valo.read(tf.shape(val)[0]-1) 
		outputs.append(tf.expand_dims(last,1))

	# attention mechanism over different categories of objects
	encoded = tf.concat(outputs, 1) 
	state = att_stacked_lstm.zero_state(batch_size, tf.float32) 
	print(encoded.get_shape().as_list(), state[0].get_shape().as_list())
	for cat in range(categories):
		a = tf.nn.softmax(tf.einsum("ijk,ik->ij", encoded, state[0])) 
		a_sum = tf.reduce_sum(tf.expand_dims(a,-1)*encoded, axis=1) 
		with tf.variable_scope('lstm', reuse= cat>0):
			output, state = att_stacked_lstm(a_sum, state)

	final_state = output

	return final_state, lstm1



np.random.seed(0)
tf.set_random_seed(1234)
#	categories ['rbird', 'bbird', 'pig', 'wood', 'ice', 'stone']
input_st = [np.random.rand(batch_size, 16, n_coord), np.random.rand(batch_size, 1, n_coord), np.random.rand(batch_size, 5, n_coord), np.random.rand(batch_size, 11, n_coord),\
		np.random.rand(batch_size, 5, n_coord), np.random.rand(batch_size, 1, n_coord)]

in_state = [tf.placeholder(tf.float32, shape=(batch_size, None, n_coord)) for _ in xrange(categories)]
final_state, lstm1  = encode(in_state)

cost = tf.reduce_mean(tf.square(tf.ones([batch_size, n_hidden]) - final_state))
optimizer = tf.train.AdamOptimizer(1).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	with tf.variable_scope('weights', reuse=True):
              w_h2h = tf.get_variable('w_b')
	with tf.variable_scope('lstm/rnn/multi_rnn_cell/cell_1/basic_lstm_cell', reuse=True):
		w_h2h2 = tf.get_variable('w_b')
	out,w1 = sess.run([final_state,w_h2h],feed_dict={i: d for i, d in zip(in_state, input_st)})
	print(out)
	print(out.shape)
	print(w1)
	sess.run(optimizer,feed_dict={i: d for i, d in zip(in_state, input_st)})
	
	wls = lstm1.return_weights()[2]
	w1, w2 = sess.run([w_h2h, wls],feed_dict={i: d for i, d in zip(in_state, input_st)})
	print(w1, w2)

	for var in tf.trainable_variables():
		print(var.name)



"""
batch_size = 3
n_hidden = state_dim = 3
action_dim = 2
categories = 6
n_coord = 5
n_layers = 2 
np.random.seed(0)
tf.set_random_seed(1234)
#	categories ['rbird', 'bbird', 'pig', 'wood', 'ice', 'stone']
input_st = [np.random.rand(batch_size, 16, n_coord), np.random.rand(batch_size, 1, n_coord), np.random.rand(batch_size, 5, n_coord), np.random.rand(batch_size, 11, n_coord),\
		np.random.rand(batch_size, 5, n_coord), np.random.rand(batch_size, 1, n_coord)]

encoder = StateEnc()
in_state = [tf.placeholder(tf.float32, shape=(batch_size, None, n_coord)) for _ in xrange(categories)]
final_state  = encoder.encode(in_state)

cost = tf.reduce_mean(tf.square(tf.zeros([batch_size, n_hidden]) - final_state))
optimizer = tf.train.AdamOptimizer(0.5).minimize(cost)
grad_and_vars = tf.train.AdamOptimizer(0.5).compute_gradients(cost)
train_op = tf.train.AdamOptimizer(0.5).apply_gradients(grad_and_vars)

with tf.Session() as sess:
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	
	weights, biases, w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2 = encoder.get_parameters()
	out,w1 = sess.run([final_state,w_i2h1],feed_dict={i: d for i, d in zip(in_state, input_st)})
	print(out)
	print(out.shape)
	print(w1)
	weights, biases, w_i2h0, w_h2h0, w_b0, w_i2h1, w_h2h1, w_b1, w_i2h2, w_h2h2, w_b2 = encoder.get_parameters()
	_,w1,w2 = sess.run([train_op,w_i2h1,w_i2h2],feed_dict={i: d for i, d in zip(in_state, input_st)})
	print(w1, w2)

	for g, v in grad_and_vars:
          print(v.name,v.device, g) 
"""