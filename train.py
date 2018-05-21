import os
import sys
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

# Ignore DeprecationWarning
import warnings
warnings.filterwarnings('ignore')

def main():
	parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
	parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of sweeps over the dataset to train')
	args = parser.parse_args()
	# Load dataset
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

	test_x = np.array(states[:test_data_size].copy())
	train_x = np.array(states[test_data_size:].copy())
	del states
	test_y = np.array(actions[:test_data_size].copy())
	train_y = np.array(actions[test_data_size:].copy())
	del actions

	model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
	model.to_gpu()
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

	# 学習ループ
	for epoch in tqdm(range(args.epoch)):
		start = time.time()
		shuffle = np.random.permutation(train_data_size)
		for idx in tqdm(range(0, train_data_size, minibatch_size)):
			x = train_x[shuffle[idx:min(idx+minibatch_size, train_data_size)]]
			x = chainer.Variable(cuda.to_gpu(x.reshape(-1, 1, 8, 8).astype(np.float32)))
			y = train_y[shuffle[idx:min(idx+minibatch_size, train_data_size)]]
			y = chainer.Variable(cuda.to_gpu(y.astype(np.int32)))
			optimizer.update(model, x, y)
			#sys.stdout.write("*")
		#sys.stdout.flush()
	    # Test
		idx = np.random.choice(test_x.shape[0], val_size, replace=False)
		x = chainer.Variable(cuda.to_gpu(test_x[idx].reshape(-1, 1, 8, 8).astype(np.float32)))
		y = chainer.Variable(cuda.to_gpu(test_y[idx].astype(np.int32)))
		print('\nepoch :', epoch, '  loss :', model(x, y).data)
		elapsed_time = time.time() - start
		print(str((epoch+1)/args.epoch*100) + '% done.')
		delta = elapsed_time*(args.epoch-epoch-1)
		delta = timedelta(seconds=+delta)
		fin = datetime.now() + delta
		fin = fin.strftime("%Y-%m-%d %H:%M")
		print("Estimated to finish at " + fin)
		serializers.save_npz('model.npz', model)  # モデル保存
		serializers.save_npz('optimizer.npz', optimizer)  # モデル保存

if __name__ == '__main__':
    main()
