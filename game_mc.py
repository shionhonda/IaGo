import numpy as np
import random
import copy
import time
import chainer
import chainer.links as L
from chainer import Variable, serializers
import policy
import mcts_self_play
from numba import jit

class Game:

    def __init__(self):
        # Initialize board state
        self.state = np.zeros([8, 8], dtype=np.float32)
        self.state[4, 3] = 1
        self.state[3, 4] = 1
        self.state[3, 3] = 2
        self.state[4, 4] = 2
        # Initialize game variables
        self.stone_num = 4
        self.pass_flg = False
        # Initialize model
        self.model = L.Classifier(policy.SLPolicy())
        serializers.load_npz('./models/sl_model.npz', self.model)
        

    # Place a stone and turn all the sandwithced stones
    # Position y:vertical, x:horizontal
    # Color  1:white, 2:black
    @jit
    def place_stone(self, position, color):
        # Place the stone
        pos = np.array(position)-[1,1]
        self.state[pos[0], pos[1]] = color
        # Search for sandwitched stones
        dys = [-1, -1, -1, 0, 0, 1, 1, 1] # Search direction
        dxs = [-1, 0, 1, -1, 1, -1, 0, 1] # Search direction
        for dy,dx in zip(dys, dxs):
            if is_outside(pos+[dy,dx]):
                continue # Search next direction if index goes out of range
            if self.state[pos[0]+dy, pos[1]+dx]+color!=3:
                continue # Search next direction if empty or same-color stone
            ref = pos + [dy, dx] # Decide direction
            while(self.state[ref[0], ref[1]]+color==3):
                ref += [dy, dx] # Referring index
                if is_outside(ref):
                    break # Stop if referring index goes out of range
            if is_outside(ref):
                continue # Search next direction if index goes out of range
            # Turn sandwitched stones
            if self.state[ref[0], ref[1]]==color:
                ref -= [dy, dx]
                while(self.state[ref[0], ref[1]]+color==3):
                    self.state[ref[0], ref[1]] = 3-self.state[ref[0], ref[1]]
                    ref -= [dy, dx]

    def valid_pos(self, color):
        positions = []
        for i in range(8):
            for j in range(8):
                if self.state[i, j] != 0:
                    continue
                # Search 8 directions
                dys = [-1, -1, -1, 0, 0, 1, 1, 1]
                dxs = [-1, 0, 1, -1, 1, -1, 0, 1]
                for dy,dx in zip(dys, dxs):
                    if is_outside([i+dy, j+dx]):
                        continue
                    if self.state[i+dy, j+dx]+color!=3:
                        continue
                    ref = np.array([i+dy, j+dx])
                    while(self.state[ref[0], ref[1]]+color==3):
                        ref += [dy, dx]
                        out_flg = is_outside(ref)
                        if out_flg:
                            break # Stop if referring index goes out of range
                    if out_flg:
                        continue
                    if self.state[ref[0], ref[1]]==color:
                        positions.append([i+1,j+1])
                        break
        return positions

    # Judge game winner
    def judge(self):
        you = np.sum(self.state==1)
        ai = np.sum(self.state==2)
        if you>ai:
            return 1
        elif you<ai:
            return -1
        else:
            return 0

    # Get position to place stone
    def get_position(self, color, positions):
        state = copy.deepcopy(self.state)
        #print(state)
        if color==1:
            # AI1's turn
            # Monte Carlo

            if len(positions)>1:
                expects = np.zeros(len(positions))
                for i in range(len(positions)):
                    for g in range(20):
                        sim = mcts_self_play.Simulate(state)
                        expects[i] += sim(positions[i])
                #print(positions)
                #print(expects)
                position = positions[np.argmax(expects)]
            else:
                position = positions[0]

        else:
            # Predict position to place stone
            X = np.stack([state==1, state==2], axis=0).astype(np.float32)
            state_var = chainer.Variable(X.reshape(2,1,8,8).transpose(1,0,2,3))
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    action_probabilities = self.model.predictor(state_var).data.reshape(64)
            idx = np.random.choice(64, p=softmax(action_probabilities))
            position = [idx//8+1, idx%8+1]
            if not position in positions:
                # Choose again if prediction is illegal
                return self.get_position(color, positions)
                #return random.choice(positions)
        #print(position, "\n")
        return position

    # Things to do in one turn
    def turn(self, color):
        positions = self.valid_pos(color)
        if len(positions)>0:
            position = self.get_position(color, positions)
            self.place_stone(position, color)
            self.pass_flg = False
            self.stone_num += 1
        else:
            if self.pass_flg:
                self.stone_num = 64 # Game over when two players pass consecutively
            self.pass_flg = True

# Return True if the index is out of the board
@jit
def is_outside(pos):
    return pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)

def main():
    result = np.zeros(3)
    tm = 0
    for num in range(50):
        start = time.time()
        game = Game()
        # Whole game
        # def __call__(self):
        while(game.stone_num<64):
            game.turn(1)
            game.turn(2)
        jd = game.judge()
        print("result", jd)
        result[jd+1] += 1
        #print(game.state)
        elapsed_time = time.time() - start
        tm += elapsed_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print(result)
    print(tm/50, "sec")


if __name__ == '__main__':
    main()
