import argparse
import numpy as np
from tqdm import tqdm
import value_self_play

def main():
    # Set the number of sets
     parser = argparse.ArgumentParser(description='IaGo:')
     parser.add_argument('--size', '-s', type=int, default=1000000, help='Number of games to play')
     args = parser.parse_args()

     for i in tqdm(range(args.size)):
        rand = np.random.randint(4,64)
        self_play = value_self_play.SelfPlay(rand)
        state, result = self_play()
        with open("./value_data5.txt", "a") as f:
            f.write("\n"+str(state) + ", \r")
            f.write("\n")
            f.write("\n"+str(result) + ", \r")

if __name__ == '__main__':
    main()
