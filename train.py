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
	parser.add_argument('--epoch', '-e', type=int, default=50, help='Number of sweeps over the dataset to train')
	args = parser.parse_args()

	# Model definition
	model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
	model.to_gpu()
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
	print("Model set complete.")

	# Learing loop
	for epoch in tqdm(range(args.epoch)):
		# Load dataset ans set parameters
		states = np.load('data/states.npy')
		actions = np.load('data/actions.npy')
		minibatch_size = 1024
		data_size = len(actions)
		test_data_size = 1000
		val_size = 1000
		small_size = 8001000
		train_size = small_size - val_size

		# Shuffle dataset
		rands = np.random.choice(data_size, small_size, replace=False)
		states = [states[rands[i],:,:] for i in range(small_size)]
		actions = [actions[rands[i]] for i in range(small_size)]

		# Devide dataset into train and test
		test_x = np.array(states[:test_data_size].copy())
		train_x = np.array(states[test_data_size:].copy())
		del states
		test_y = np.array(actions[:test_data_size].copy())
		train_y = np.array(actions[test_data_size:].copy())
		del actions

		# Minibatch learning
		shuffle = np.random.permutation(train_size)
		for idx in tqdm(range(0, train_size, minibatch_size)):
			x = train_x[shuffle[idx:min(idx+minibatch_size, train_size)]]
			X = np.stack([x==1, x==2], axis=3)
			X = chainer.Variable(cuda.to_gpu(X.reshape(-1, 2, 8, 8).astype(np.float32)))
			y = train_y[shuffle[idx:min(idx+minibatch_size, train_size)]]
			y = chainer.Variable(cuda.to_gpu(y.astype(np.int32)))
			optimizer.update(model, X, y)
		# Test
		idx = np.random.choice(test_x.shape[0], val_size, replace=False)
		X = np.stack([test_x[idx]==1, test_x[idx]==2], axis=3)
		X = chainer.Variable(cuda.to_gpu(X.reshape(-1, 2, 8, 8).astype(np.float32)))
		y = chainer.Variable(cuda.to_gpu(test_y[idx].astype(np.int32)))
		print('\nepoch :', epoch, '  loss :', model(X, y).data)
		# Save models
		serializers.save_npz('model.npz', model)
		serializers.save_npz('optimizer.npz', optimizer)

if __name__ == '__main__':
    main()
