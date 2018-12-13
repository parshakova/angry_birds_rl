class Hp:
	batch_size = 64
	n_hidden = state_dim = 100; epochs = 1000000; action_dim = 3
	#redB, yelB, bluB, pig, ice, wood, stone
	categories = 7
	n_coord = 4 
	n_layers = 2 # num of layers if 1st lstm of encoder
	input_proj = 50; N_TRAIN = 2
	REPLAY_BUFFER_SIZE = 1000000
	REPLAY_START_SIZE = 500
	GAMMA = 0.99
	# actor
	aLAYER1_SIZE = 400
	aLAYER2_SIZE = 300
	aLEARNING_RATE = 1e-3
	TAU = 0.001
	#critic
	cLAYER1_SIZE = 400
	cLAYER2_SIZE = 300
	cLEARNING_RATE = 1e-2
	cL2 = 0.01
