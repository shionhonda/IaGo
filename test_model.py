import argparse
import chainer
import chainer.links as L
from chainer import serializers, optimizers
import SLPolicy
import rl_self_play

def main():
	# Set the number of episodes
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--episode', '-e', type=int, default=32, help='Number of games to play')
    args = parser.parse_args()

    # Model definition
    model1 = L.Classifier(SLPolicy.SLPolicyNet())
    serializers.load_npz("./models/rl/model49.npz", model1)
    optimizer = optimizers.Adam()
    optimizer.setup(model1)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
    chainer.backends.cuda.get_device_from_id(0).use()
    model1.to_gpu()
    model2 = L.Classifier(SLPolicy.SLPolicyNet())
    serializers.load_npz("./models/rl/model0.npz", model2)
    model2.to_gpu()

    for i in range(args.episode):
        result = 0
        if i%2==0:
            game = rl_self_play.Game(model1, model2)
            reward = game()
        else:
            # Switch head and tail
            game = rl_self_play.Game(model2, model1)
            reward = -game()
        result += reward
        print(reward)
    print("Result:", result)

if __name__ == '__main__':
    main()
