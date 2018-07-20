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
	parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train')
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

	test_x = np.load('data/states_test.npy')
	test_y = np.load('data/actions_test.npy')
	test_x, test_y = transform(test_x, test_y)
	# Load train dataset
	train_x = np.load('data/states.npy')
	train_y = np.load('data/actions.npy')
	train_size = train_y.shape[0]
	minibatch_size = 4096 # 2**12

	# Learing loop
	for epoch in tqdm(range(args.epoch)):
		model.to_gpu(args.gpu)
		# Shuffle train dataset
		rands = np.random.choice(train_size, train_size, replace=False)
		train_x = train_x[rands,:,:]
		train_y = train_y[rands]

		# Minibatch learning
		for idx in tqdm(range(0, train_size, minibatch_size)):
			x = train_x[idx:min(idx+minibatch_size, train_size), :, :]
			y = train_y[idx:min(idx+minibatch_size, train_size)]
			x, y = transform(x, y)
			train_pred = model(x)
			train_loss = F.softmax_cross_entropy(train_pred, y)
			model.cleargrads()
			train_loss.backward()
			optimizer.update()
		# Calculate loss
		with chainer.using_config('train', False):
			with chainer.using_config('enable_backprop', False):
				test_pred = model(test_x)
		test_loss = F.softmax_cross_entropy(test_pred, test_y).data
		test_acc = F.accuracy(test_pred, test_y).data
		print('\nepoch :', epoch, '  loss :', test_loss, ' accuracy:', test_acc)
		# Log
		if args.policy=="rollout":
			with open("./log_rollout.txt", "a") as f:
				f.write(str(test_loss) + " : " + str(test_acc) + ", \n")
		else:
			with open("./log_sl.txt", "a") as f:
				f.write(str(test_loss) + " : " + str(test_acc) + ", \n")
		# Save models
		model.to_cpu()
		if args.policy=="rollout":
			serializers.save_npz('./models/rollout_model.npz', model)
			serializers.save_npz('./models/rollout_optimizer.npz', optimizer)
		else:
			serializers.save_npz('sl_model.npz', model)
			serializers.save_npz('sl_optimizer.npz', optimizer)
		# Early stop
		#if loss<0.94:
		#	print("Early stop")
		#	break

if __name__ == '__main__':
    main()
