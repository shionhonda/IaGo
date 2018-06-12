import argparse
import glob
import numpy as np
from tqdm import tqdm
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, cuda, optimizers, Variable
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
import chainerrl
import SLPolicy
import rl_self_play

def main():
	# Set the number of episodes
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--set', '-s', type=int, default=1, help='Number of game sets played to train')
    args = parser.parse_args()
    N = 10

    # Model definition
    '''lossfunの定義がわからない'''
    model1 = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
    serializers.load_npz("./models/sl_model.npz", model1)
    optimizer = optimizers.Adam()
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))


    for set in tqdm(range(args.set)):
        model1.to_gpu()
        Z = np.zeros(2*N)
        for i in tqdm(range(N)):
            # Randomly choose competitor model from reinforced models
            model2 = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
            model2_path = np.random.choice(glob.glob("./models/rl/*.npz"))
            serializers.load_npz(model2_path, model2)
            model2.to_gpu()
            game = rl_self_play.Game(model1, model2)
            Z[i] = game()
            # Switch head and tail
            game = rl_self_play.Game(model2, model1)
            Z[i+N] = -game()
        print(Z)
        '''Update model1'''
        if set%50==0:
            serializers.save_npz("./models/rl/"+str(set//50)+".npz")

if __name__ == '__main__':
    main()
