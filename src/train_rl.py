import argparse
import glob
import numpy as np
import random
import copy
from tqdm import tqdm
import chainer
import chainer.functions as F
from chainer import serializers, optimizers, Variable
import network
import rl_self_play

def main():
    # Set the number of sets
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--models', '-m', type=int, default=1, help='Number of trained models')
    parser.add_argument('--set', '-s', type=int, default=1000, help='Number of game sets played to train')
    args = parser.parse_args()
    N = 32

    # Model definition
    model1 = network.SLPolicy()
    serializers.load_npz("../models/RL/model2.npz", model1)
    optimizer = optimizers.Adam()
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
    serializers.load_npz("../models/RL/optimizers/2.npz", optimizer)
    # REINFORCE algorithm
    models = args.models
    cnt = 0
    #for set in tqdm(range(0, args.set)):
    while(models<=20):
        # Randomly choose competitor model from reinforced models
        model2 = network.SLPolicy()
        model2_path = np.random.choice(glob.glob("../models/RL/*.npz"))
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
        x = Variable(x.transpose(1,0,2,3))
        y = Variable(np.array(action_seq).astype(np.int32))
        r = Variable(np.array(reward_seq).astype(np.float32))
        pred = model1(x)
        c  = F.softmax_cross_entropy(pred, y, reduce="no")
        model1.cleargrads()
        loss = F.mean(c*r)
        loss.backward()
        optimizer.update()
        rate = result/(2*N)
        print("Models:" + str(models) + ", Result:" + str(rate) + ", Loss:" + str(loss.data))
        with open("../log/rl.txt", "a") as f:
            f.write(str(rate) + ", \n")
        if rate>0.5:
            cnt += 1
        if cnt>4*np.sqrt(models) and rate>0.6:
            model = copy.deepcopy(model1)
                #model.to_cpu()
            serializers.save_npz("../models/RL/model"+str(models)+".npz", model)
            serializers.save_npz("../models/RL/optimizers/"+str(models)+".npz", optimizer)
            models += 1
            cnt = 0
        if rate<0.2:
            break


        #if (set+1)%50==0:
        #    model = copy.deepcopy(model1)
            #model.to_cpu()
        #    serializers.save_npz("../models/RL/model"+str((set+1)//50)+".npz", model)
        #    serializers.save_npz("../models/rl_optimizer.npz", optimizer)

if __name__ == '__main__':
    main()
