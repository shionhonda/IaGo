import argparse
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
import rl_env

def main():
	# Set the number of episodes
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--set', '-s', type=int, default=1000, help='Number of game sets played to train')
    args = parser.parse_args()
    N = 32

    # Model definition
    model1 = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
    serializers.load_npz("./models/rl/model0.npz", model1)
    optimizer = optimizers.Adam()
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
    model2 = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
    serializers.load_npz("./models/rl/model0.npz", model2)
    # REINFORCE algorithm
    agent = chainerrl.agents.REINFORCE(model1, optimizer, batchsize=2*N,
    backward_separately=True)

    env = rl_env.GameEnv(agent.model, model2)
    for set in tqdm(range(args.set)):
        result = 0
        #b = np.array([]) # Baseline

        for i in range(2*N):
            obs = env.reset()
            reward = 0
            done = False
            while not done:
                action = agent.act_and_train(obs, reward)
                obs, reward, done, _ = env.step(action)
            judge = env()
            #b = np.append(b, judge)
            agent.reward_sequences[-1] = [judge]*len(agent.log_prob_sequences[-1])
            if judge==1:
                result += 1
            # Update model if i reaches batchsize 2*N
            agent.stop_episode_and_train(obs, judge, done=True)
        print("\nSet:" + str(set) + ", Result:" + str(result/(2*N)))
        with open("./log_test.txt", "a") as f:
            f.write(str(result/(2*N))+", \n")

        if (set+1)%10==0:
            model = copy.deepcopy(agent.model)
            #model.to_cpu()
            serializers.save_npz("./models/rl/test_model.npz", model)
            serializers.save_npz("./models/test_optimizer.npz", agent.optimizer)

if __name__ == '__main__':
    main()
