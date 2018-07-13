import argparse
import glob
import numpy as np
import random
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
import rl_env

def main():
	# Set the number of sets
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--set', '-s', type=int, default=1000, help='Number of game sets played to train')
    args = parser.parse_args()
    N = 32

    # Model definition
    model1 = L.Classifier(SLPolicy.SLPolicyNet())
    serializers.load_npz("./models/rl/model0.npz", model1)
    optimizer = optimizers.Adam()
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    # REINFORCE algorithm
    agent = chainerrl.agents.MyREINFORCE(model1, optimizer, batchsize=2*N,
    backward_separately=True)

    for set in tqdm(range(0, args.set)):
        # Randomly choose competitor model from reinforced models
        model2 = L.Classifier(SLPolicy.SLPolicyNet())
        model2_path = np.random.choice(glob.glob("./models/rl/*.npz"))
        print(model2_path)
        serializers.load_npz(model2_path, model2)
        env = rl_env.GameEnv(agent.model, model2)
        result = 0
        for i in  tqdm(range(2*N)):
            obs = env.reset()
            if i%2==1:
                # Switch head and tail
                pos = random.choice([[2,4], [3,5], [4,2], [5,3]])
                env.state[pos[0], pos[1]] = 2

            X = np.stack([env.state==1, env.state==2], axis=0).astype(np.float32)
            obs = chainer.Variable(X.reshape(2,1,8,8).transpose(1,0,2,3))
            reward = 0
            done = False
            while not done:
                action = agent.act_and_train(obs, reward)
                obs, reward, done, _ = env.step(action)
            judge = env()
            agent.reward_sequences[-1] = [judge]*len(agent.log_prob_sequences[-1])
            if judge==1:
                result += 1
            # Update model if i reaches batchsize 2*N
            agent.stop_episode_and_train(obs, judge, done=True)
        print("\nSet:" + str(set) + ", Result:" + str(result/(2*N)))
        with open("./log_rl.txt", "a") as f:
            f.write(str(result/(2*N))+", \n")

        model = copy.deepcopy(agent.model)
            #model.to_cpu()
        serializers.save_npz("./backup/model"+str(set)+".npz", model)
        serializers.save_npz("./backup/rl_optimizer.npz", agent.optimizer)


        if (set+1)%20==0:
            #model = copy.deepcopy(agent.model)
            #model.to_cpu()
            serializers.save_npz("./models/rl/model"+str((set+1)//20)+".npz", model)
            serializers.save_npz("./models/rl_optimizer.npz", agent.optimizer)

if __name__ == '__main__':
    main()
