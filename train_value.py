import argparse
import numpy as np
from tqdm import tqdm

import chainer
import chainer.links as L
from chainer import serializers, cuda, optimizers, Variable
from chainer.functions import mean_squared_error

import value

def main():
	# Set the number of epochs
	parser = argparse.ArgumentParser(description='IaGo:')
	parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of sweeps over the dataset to train')
	args = parser.parse_args()

	# Model definition
	model = L.Classifier(value.ValueNet(), lossfun=mean_squared_error)
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

	test_x = np.load('value_data/npy/states2.npy')
	test_y = np.load('value_data/npy/results2.npy')
	test_x = np.stack([test_x==1, test_x==2], axis=0).astype(np.float32)
	test_x = chainer.Variable(cuda.to_gpu(test_x.transpose(1,0,2,3)))
	test_y = chainer.Variable(cuda.to_gpu(test_y.astype(np.float32)))
	# Load train dataset
	train_x = np.load('value_data/npy/states1.npy')
	train_y = np.load('value_data/npy/results1.npy')
	train_size = train_y.shape[0]
	minibatch_size = 4096 # 2**12

	# Learing loop
	for epoch in tqdm(range(args.epoch)):
		model.to_gpu()
		# Shuffle train dataset
		rands = np.random.choice(train_size, train_size, replace=False)
		train_x = train_x[rands,:,:]
		train_y = train_y[rands]

		# Minibatch learning
		for idx in tqdm(range(0, train_size, minibatch_size)):
			x = train_x[idx:min(idx+minibatch_size, train_size), :, :]
			x = np.stack([x==1, x==2], axis=0).astype(np.float32)
			x = chainer.Variable(cuda.to_gpu(x.transpose(1,0,2,3)))
			y = train_y[idx:min(idx+minibatch_size, train_size)]
			y = chainer.Variable(cuda.to_gpu(y.astype(np.float32)))
			pred = model.predictor(x)
			loss = mean_squared_error(pred, y)
			model.cleargrads()
			loss.backward
			optimizer.update()
		# Calculate loss
		loss = model(test_x, test_y).data
		print('\nepoch :', epoch, '  loss :', loss)
		# Log
		with open("./log_value.txt", "a") as f:
			f.write(str(loss)+", \n")
		# Save models
		model.to_cpu()
		serializers.save_npz('models/value_model.npz', model)
		serializers.save_npz('models/value_optimizer.npz', optimizer)
		# Early stop
		#if loss<0.94:
		#	print("Early stop")
		#	break

if __name__ == '__main__':
    main()
