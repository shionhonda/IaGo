import argparse
import glob
import numpy as np
import random
import copy
from tqdm import tqdm
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, optimizers, Variable
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
import network
import rl_self_play

def main():
	# Set the number of sets
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--set', '-s', type=int, default=10000, help='Number of game sets played to train')
    args = parser.parse_args()
    N = 32

    # Model definition
    model1 = network.SLPolicy()
    serializers.load_npz("../models/RL/model0.npz", model1)
    optimizer = optimizers.Adam(alpha=0.0005)
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
    #serializers.load_npz("./backup/rl_optimizer.npz", optimizer)
    # REINFORCE algorithm

    for set in tqdm(range(0, args.set)):
        # Randomly choose competitor model from reinforced models
        model2 = network.SLPolicy()
        model2_path = np.random.choice(glob.glob("../models/RL/model0.npz"))
        print(model2_path)
        serializers.load_npz(model2_path, model2)

        result = 0
        state_seq, action_seq, reward_seq = [], [], []
        for i in tqdm(range(2*N)):
            game = rl_self_play.Game(model1, model2)
            if i%2==1:
                # Switch head and tail
                pos = random.choice([[2,4], [3,5], [4,2], [5,3]])
                game.state[pos[0], pos[1]] = 2
            states, actions, judge = game()
            rewards = [judge]*len(states)
            state_seq += states
            action_seq += actions
            reward_seq += rewards
            if judge==1:
                result += 1

        # Update model
        x = np.array(state_seq)
        x = np.stack([x==1, x==2], axis=0).astype(np.float32)
        x = chainer.Variable(x.transpose(1,0,2,3))
        y = Variable(np.array(action_seq).astype(np.int32))
        r = Variable(np.array(reward_seq).astype(np.float32))
        pred = model1(x)
        c  = softmax_cross_entropy(pred, y, reduce="no")
        model1.cleargrads()
        loss = F.mean(c*r)
        loss.backward()
        optimizer.update()
        print("Set:" + str(set) + ", Result:" + str(result/(2*N)) + ", Loss:" + str(loss.data))
        with open("./log_test.txt", "a") as f:
            f.write(str(result/(2*N)) + ", \n")

        model = copy.deepcopy(model1)
            #model.to_cpu()
        #serializers.save_npz("./backup/model"+str(set)+".npz", model)
        #serializers.save_npz("./backup/optimizer"+str(set)+".npz", optimizer)


        if (set+1)%500==0:
            model = copy.deepcopy(model1)
            #model.to_cpu()
            serializers.save_npz("../models/RL/model"+str((set+1)//500)+".npz", model)
            serializers.save_npz("../models/rl_optimizer.npz", optimizer)

if __name__ == '__main__':
    main()
