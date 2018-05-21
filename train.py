import argparse
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm

import chainer
import chainer.links as L
from chainer import training, serializers, cuda, optimizers, Variable
from chainer.training import extensions
from chainer.training import triggers
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import SLPolicy

def main():
	# Set the number of epochs
	parser = argparse.ArgumentParser(description='IaGo:')
	parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train')
	args = parser.parse_args()
	# Load dataset ans set parameters
	states = np.load('data/states.npy')
	actions = np.load('data/actions.npy')
	minibatch_size = 512
	data_size = len(actions)
	train_data_size = int(data_size*4/5)
	test_data_size = int(data_size/5)
	val_size = 1000

	# Shuffle dataset
	rands = np.random.choice(data_size, data_size, replace=False)
	states = [states[rands[i],:,:] for i in range(data_size)]
	actions = [actions[rands[i]] for i in range(data_size)]
	# Devide dataset into train and test
	test_x = np.array(states[:test_data_size].copy())
	train_x = np.array(states[test_data_size:].copy())
	del states
	test_y = np.array(actions[:test_data_size].copy())
	train_y = np.array(actions[test_data_size:].copy())
	del actions
	# Model definition
	model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
	model.to_gpu()
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

	# Learing loop
	for epoch in tqdm(range(args.epoch)):
		# Minibatch learning
		shuffle = np.random.permutation(train_data_size)
		for idx in tqdm(range(0, train_data_size, minibatch_size)):
			x = train_x[shuffle[idx:min(idx+minibatch_size, train_data_size)]]
			x = chainer.Variable(cuda.to_gpu(x.reshape(-1, 1, 8, 8).astype(np.float32)))
			y = train_y[shuffle[idx:min(idx+minibatch_size, train_data_size)]]
			y = chainer.Variable(cuda.to_gpu(y.astype(np.int32)))
			optimizer.update(model, x, y)
	    # Test
		idx = np.random.choice(test_x.shape[0], val_size, replace=False)
		x = chainer.Variable(cuda.to_gpu(test_x[idx].reshape(-1, 1, 8, 8).astype(np.float32)))
		y = chainer.Variable(cuda.to_gpu(test_y[idx].astype(np.int32)))
		print('\nepoch :', epoch, '  loss :', model(x, y).data)
		# Save models
		serializers.save_npz('model.npz', model)  
		serializers.save_npz('optimizer.npz', optimizer)

if __name__ == '__main__':
    main()
