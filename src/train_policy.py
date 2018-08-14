import argparse
import numpy as np
from tqdm import tqdm
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, cuda, optimizers, Variable
import network

def transform(x, y):
	x = np.stack([x==1, x==2], axis=0).astype(np.float32)
	X = Variable(cuda.to_gpu(x.transpose(1,0,2,3)))
	Y = Variable(cuda.to_gpu(y.astype(np.int32)))
	return X, Y

def main():
	# Set the number of epochs and policy to train
	parser = argparse.ArgumentParser(description='IaGo:')
	parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of sweeps over the dataset to train')
	parser.add_argument('--policy', '-p', type=str, default="sl", help='Policy to train: sl or rollout')
	parser.add_argument('--gpu', '-g', type=int, default="0", help='GPU ID')
	args = parser.parse_args()

	# Model definition
	if args.policy=="rollout":
		model = network.RolloutPolicy()
	else:
		if args.policy!="sl":
			print('Argument "--policy" is invalid. SLPolicy has been set by default.')
		model = network.SLPolicy()
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
	cuda.get_device(args.gpu).use()

	X_test = np.load('../policy_data/npy/states_test.npy')
	y_test = np.load('../policy_data/npy/actions_test.npy')
	X_test, y_test = transform(X_test, y_test)
	# Load train dataset
	X_train = np.load('../policy_data/npy/states.npy')
	y_train = np.load('../policy_data/npy/actions.npy')
	train_size = y_train.shape[0]
	minibatch_size = 4096 # 2**12

	# Learing loop
	for epoch in tqdm(range(args.epoch)):
		model.to_gpu(args.gpu)
		# Shuffle train dataset
		rands = np.random.choice(train_size, train_size, replace=False)
		X_train = X_train[rands,:,:]
		y_train = y_train[rands]

		# Minibatch learning
		for idx in tqdm(range(0, train_size, minibatch_size)):
			x = X_train[idx:min(idx+minibatch_size, train_size), :, :]
			y = y_train[idx:min(idx+minibatch_size, train_size)]
			x, y = transform(x, y)
			pred_train = model(x)
			loss_train = F.softmax_cross_entropy(pred_train, y)
			model.cleargrads()
			loss_train.backward()
			optimizer.update()
		# Calculate loss
		with chainer.using_config('train', False):
			with chainer.using_config('enable_backprop', False):
				pred_test = model(X_test)
		loss_test = F.softmax_cross_entropy(pred_test, y_test).data
		test_acc = F.accuracy(pred_test, y_test).data
		print('\nepoch :', epoch, '  loss :', loss_test, ' accuracy:', test_acc)
		# Log
		if args.policy=="rollout":
			with open("../log/rollout.txt", "a") as f:
				f.write(str(loss_test) + ", " + str(test_acc) + "\n")
		else:
			with open("../log/sl.txt", "a") as f:
				f.write(str(loss_test) + ", " + str(test_acc) + "\n")
		# Save models
		model.to_cpu()
		if args.policy=="rollout":
			serializers.save_npz('../models/rollout_model.npz', model)
			serializers.save_npz('../models/rollout_optimizer.npz', optimizer)
		else:
			serializers.save_npz('../models/sl_model.npz', model)
			serializers.save_npz('../models/sl_optimizer.npz', optimizer)

if __name__ == '__main__':
    main()
