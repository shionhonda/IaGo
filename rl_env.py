import numpy as np
import random
import chainer
import chainer.links as L
from chainer import cuda, Variable
from chainer.cuda import cupy as cp
import sys
sys.setrecursionlimit(10000)

class GameEnv:

    def __init__(self, model1, model2):
        # Initialize board state
        self.state = cp.zeros([8, 8], dtype=np.float32)
        self.state[4, 3] = 1
        self.state[3, 4] = 1
        self.state[3, 3] = 2
        self.state[4, 4] = 2
        # Initialize game variables
        self.stone_num = 4
        self.pass_flg = False
        # Initialize model
        self.model1 = model1
        self.model2 = model2

    def reset(self):
        # Initialize board state
        self.state = np.zeros([8, 8], dtype=np.float32)
        self.state[4, 3] = 1
        self.state[3, 4] = 1
        self.state[3, 3] = 2
        self.state[4, 4] = 2
        # Initialize game variables
        self.stone_num = 4
        self.pass_flg = False

        X = np.stack([self.state==1, self.state==2], axis=0).astype(np.float32)
        obs = chainer.Variable(X.reshape(2,1,8,8).transpose(1,0,2,3))
        return obs

    def step(self, action):
        done = False
        positions = self.valid_pos(1)
        if len(positions)>0:
            position = [action//8+1, action%8+1]
            if not position in positions:
                # Choose randomly if prediction is illegal (very rare)
                position = random.choice(positions)
            self.place_stone(position, 1)
            self.stone_num += 1
            self.pass_flg = False
        else:
            if self.pass_flg:
                done = True # Game over when two players pass consecutively
            self.pass_flg = True

        # Competitor's turn
        positions = self.valid_pos(2)
        if len(positions)>0:
            position = self.get_position(2, positions)
            self.place_stone(position, 2)
            self.stone_num += 1
            self.pass_flg = False
        else:
            if self.pass_flg:
                done = True # Game over when two players pass consecutively
            self.pass_flg = True

        if self.stone_num>=64:
            done = True
        X = np.stack([self.state==1, self.state==2], axis=0).astype(np.float32)
        obs = chainer.Variable(X.reshape(2,1,8,8).transpose(1,0,2,3))

        return obs, 0, done, None


    # Whole game
    def __call__(self):
        return self.judge()

    # Return True if the index is out of the board
    def is_outside(self, pos):
        return pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7

    # Place a stone and turn all the sandwithced stones
    # Position y:vertical, x:horizontal
    # Color  1:white, 2:black
    def place_stone(self, position, color):
        # Place the stone
        pos = np.array(position)-[1,1]
        self.state[pos[0], pos[1]] = color
        # Search for sandwitched stones
        dys = [-1, -1, -1, 0, 0, 1, 1, 1] # Search direction
        dxs = [-1, 0, 1, -1, 1, -1, 0, 1] # Search direction
        for dy,dx in zip(dys, dxs):
            if self.is_outside(pos+[dy,dx]):
                continue # Search next direction if index goes out of range
            if self.state[pos[0]+dy, pos[1]+dx]+color!=3:
                continue # Search next direction if empty or same-color stone
            ref = pos + [dy, dx] # Decide direction
            while(self.state[ref[0], ref[1]]+color==3):
                ref += [dy, dx] # Referring index
                if self.is_outside(ref):
                    break # Stop if referring index goes out of range
            if self.is_outside(ref):
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
                    if self.is_outside([i+dy, j+dx]):
                        continue
                    if self.state[i+dy, j+dx]+color!=3:
                        continue
                    ref = np.array([i+dy, j+dx])
                    while(self.state[ref[0], ref[1]]+color==3):
                        ref += [dy, dx]
                        if self.is_outside(ref):
                            break
                    if self.is_outside(ref):
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
        if color==1:
            # AI1's turn
            tmp = 3*np.ones([8,8], dtype=np.float32)
            self.state = self.state*(tmp-self.state)*(tmp-self.state)/2

        # Predict position to place stone
        X = np.stack([self.state==1, self.state==2], axis=0).astype(np.float32)
        state_var = chainer.Variable(X.reshape(2,1,8,8).transpose(1,0,2,3))
        if color==1:
            action_probabilities = self.model1.predictor(state_var).data.reshape(64)
        else:
            action_probabilities = self.model2.predictor(state_var).data.reshape(64)
        #print(action_probabilities)
        action_probabilities -= np.min(action_probabilities) # Add bias to make all components non-negative
        idx = np.random.choice(64, p=action_probabilities/np.sum(action_probabilities))
        position = [idx//8+1, idx%8+1]
        if not position in positions:
            # Choose again if prediction is illegal
            return self.get_position(color, positions)
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
