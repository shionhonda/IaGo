import argparse
import numpy as np
from tqdm import tqdm

import chainer
import chainer.links as L
from chainer import serializers, cuda, optimizers, Variable
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import SLPolicy

def main():
	# Set the number of epochs
	parser = argparse.ArgumentParser(description='IaGo:')
	parser.add_argument('--epoch', '-e', type=int, default=150, help='Number of sweeps over the dataset to train')
	args = parser.parse_args()

	# Model definition
	model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

	test_x = np.load('data/states_test.npy')
	test_y = np.load('data/actions_test.npy')
	test_x = np.stack([test_x==1, test_x==2], axis=3)
	test_x = chainer.Variable(cuda.to_gpu(test_x.reshape(-1, 2, 8, 8).astype(np.float32)))
	test_y = chainer.Variable(cuda.to_gpu(test_y.reshape(-1).astype(np.int32)))
	minibatch_size = 4096 # 2**12
	train_size = 8388608 # 2**23; Approx. 1/6 of the whole data

	# Learing loop
	for epoch in tqdm(range(args.epoch)):
		model.to_gpu()
		# Load train dataset and shuffle
		train_x = np.load('data/states.npy')
		train_y = np.load('data/actions.npy')
		data_size = train_y.shape[0]
		rands = np.random.choice(data_size, train_size, replace=False)
		train_x = train_x[rands,:,:]
		train_y = train_y[rands]

		# Minibatch learning
		for idx in tqdm(range(0, train_size, minibatch_size)):
			x = train_x[idx:min(idx+minibatch_size, train_size), :, :]
			x = np.stack([x==1, x==2], axis=3)
			x = chainer.Variable(cuda.to_gpu(x.reshape(-1, 2, 8, 8).astype(np.float32)))
			y = train_y[idx:min(idx+minibatch_size, train_size)]
			y = chainer.Variable(cuda.to_gpu(y.astype(np.int32)))
			optimizer.update(model, x, y)
		# Calculate loss
		loss = model(test_x, test_y).data
		print('\nepoch :', epoch, '  loss :', loss)
		# Log
		with open("./log.txt", "a") as f:
			f.write(str(loss)+", \n")
		# Save models
		model.to_cpu()
		serializers.save_npz('models/sl_model.npz', model)
		serializers.save_npz('models/sl_optimizer.npz', optimizer)

if __name__ == '__main__':
    main()
