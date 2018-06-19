import argparse
import glob
import numpy as np
import copy
from tqdm import tqdm
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, cuda, optimizers, Variable
from chainer.cuda import cupy as cp
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
import chainerrl
import SLPolicy
import rl_self_play

def main():
	# Set the number of sets
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--set', '-s', type=int, default=1000, help='Number of game sets played to train')
    args = parser.parse_args()
    N = 16

    # Model definition
    model1 = L.Classifier(SLPolicy.SLPolicyNet())
    serializers.load_npz("./models/rl/model0.npz", model1)
    optimizer = optimizers.Adam()
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
    chainer.backends.cuda.get_device_from_id(0).use()
    model1.to_gpu()
    # REINFORCE algorithm
    agent = chainerrl.agents.REINFORCE(model1, optimizer, batchsize=2*N,
    backward_separately=False)

    for set in tqdm(range(args.set)):
        # Randomly choose competitor model from reinforced models
        model2 = L.Classifier(SLPolicy.SLPolicyNet())
        model2_path = np.random.choice(glob.glob("./models/rl/*.npz"))
        print(model2_path)
        serializers.load_npz(model2_path, model2)
        model2.to_gpu()
        result = 0
        for i in range(2*N):
            if i%2==0:
                game = rl_self_play.Game(model1, model2)
                reward = game()
            else:
                # Switch head and tail
                game = rl_self_play.Game(model2, model1)
                reward = -game()
            result += reward
            # Update model if i reaches batchsize 2*N
            X = np.array(game.states)
            X = np.stack([X==1, X==2], axis=3)
            states_var = chainer.Variable(X.reshape(-1, 2, 8, 8).astype(cp.float32))
            agent.stop_episode_and_train(states_var, reward*np.ones(states_var.shape[0]), done=False)
        print("\nSet:" + str(set) + ", Result:" + str(result))
        with open("./log_rl.txt", "a") as f:
            f.write(str(result)+", \n")

        if (set+1)%20==0:
            model = copy.deepcopy(agent.model)
            model.to_cpu()
            serializers.save_npz("./models/rl/model"+str((set+1)//20)+".npz", model)
            serializers.save_npz("./models/rl_optimizer.npz", agent.optimizer)

if __name__ == '__main__':
    main()
