import os
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer import training, serializers, cuda, optimizers, Variable
from chainer.training import extensions
from chainer.training import triggers
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import SLPolicy

def main():
	parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
	parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of sweeps over the dataset to train')
	args = parser.parse_args()

	TEST_DATA_SIZE = 100000  # テストデータのサイズ
	MINIBATCH_SIZE = 100  # ミニバッチサイズ
	EVALUATION_SIZE = 1000  # 評価のときのデータサイズ

	states = np.load('data/states.npy')
	actions = np.load('data/actions.npy')

	# 0-lengの乱数：indexに使う
	leng = len(actions)
	rands = np.random.choice(leng, leng, replace=False)
	states_shuffled = [states[rands[i],:,:] for i in range(leng)]
	actions_shuffled = [actions[rands[i]] for i in range(leng)]

	test_x = states[:TEST_DATA_SIZE].copy()  # ランダムに並び替え済み
	train_x = states[TEST_DATA_SIZE:].copy()
	del states  # メモリがもったいないので強制解放
	test_y = actions[:TEST_DATA_SIZE].copy()
	train_y = actions[TEST_DATA_SIZE:].copy()
	del actions

	model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
	model.to_gpu()
	optimizer = optimizers.Adam()
	optimizer.setup(model)

	# 学習ループ
	for epoch in range(args.epoch):
	    for i in range(100):
	        index = np.random.choice(train_x.shape[0], MINIBATCH_SIZE, replace=False)
	        x = chainer.Variable(cuda.to_gpu(train_x[index].reshape(MINIBATCH_SIZE, 1, 8, 8).astype(np.float32)))
	        t = chainer.Variable(cuda.to_gpu((train_y[index].astype(np.int32))))
	        optimizer.update(model, x, t)

	    # 評価
	    index = np.random.choice(test_x.shape[0], EVALUATION_SIZE, replace=False)
	    x = chainer.Variable(cuda.to_gpu(test_x[index].reshape(EVALUATION_SIZE, 1, 8, 8).astype(np.float32)))
	    t = chainer.Variable(cuda.to_gpu(test_y[index].astype(np.int32)))
	    print('epoch :', epoch, '  loss :', model(x, t).data)

	    serializers.save_npz('model.npz', model)  # モデル保存
	    serializers.save_npz('optimizer.npz', optimizer)  # モデル保存

if __name__ == '__main__':
    main()
