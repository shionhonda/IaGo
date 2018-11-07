import numpy as np
import random
import copy
import chainer
import chainer.links as L
from chainer import Variable

class Game:

    def __init__(self, model1, model2):
        # Initialize board state
        self.state = np.zeros([8, 8], dtype=np.float32)
        self.state[4, 3] = 1
        self.state[3, 4] = 1
        self.state[3, 3] = 2
        self.state[4, 4] = 2
        # Initialize game variables
        self.states = []
        self.actions = []
        self.stone_num = 4
        self.pass_flg = False
        # Initialize model
        self.model1 = model1
        self.model2 = model2 # Opponent

    # Whole game
    def __call__(self):
        while(self.stone_num<64):
            self.turn(1)
            self.turn(2)
        return self.states, self.actions, self.judge()

    # Place a stone and turn all the sandwithced stones
    # Position y:vertical, x:horizontal
    # Color  1:white, 2:black
    def place_stone(self, state, action, color):
        # Place the stone
        pos = np.array([action//8, action%8])
        state[pos[0], pos[1]] = color
        # Search for sandwitched stones
        dys = [-1, -1, -1, 0, 0, 1, 1, 1] # Search direction
        dxs = [-1, 0, 1, -1, 1, -1, 0, 1] # Search direction
        for dy,dx in zip(dys, dxs):
            if is_outside(pos+[dy,dx]):
                continue # Search next direction if index goes out of range
            if state[pos[0]+dy, pos[1]+dx]+color!=3:
                continue # Search next direction if empty or same-color stone
            ref = pos + [dy, dx] # Decide direction
            while(state[ref[0], ref[1]]+color==3):
                ref += [dy, dx] # Referring index
                if is_outside(ref):
                    break # Stop if referring index goes out of range
            if is_outside(ref):
                continue # Search next direction if index goes out of range
            # Turn sandwitched stones
            if state[ref[0], ref[1]]==color:
                ref -= [dy, dx]
                while(state[ref[0], ref[1]]+color==3):
                    state[ref[0], ref[1]] = color
                    ref -= [dy, dx]
        return state

    def legal_actions(self, color):
        actions = []
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
                        actions.append(i*8+j)
                        break
        return actions

    # Judge game winner
    def judge(self):
        myself = np.sum(self.state==1)
        opponent = np.sum(self.state==2)

        if myself>opponent:
            return 1
        elif myself<opponent:
            return -1
        else:
            return 0

    def make_state_var(self, state, color):
        if color==1:
            tmp = 3*np.ones([8,8], dtype=np.float32)
            state = state*(tmp-state)*(tmp-state)/2
        state_2ch = np.stack([state==1, state==2], axis=0).astype(np.float32)
        state_var = chainer.Variable(state_2ch.reshape(2,1,8,8).transpose(1,0,2,3))
        return state_var

    # Get position to place stone
    def get_action(self, color, actions):
        state_var =  self.make_state_var(self.state, color)
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                if color==1:
                    prob = self.model1(state_var).data.reshape(64)
                else:
                    prob = self.model2(state_var).data.reshape(64)
        valid = np.zeros(64)
        valid[actions] = 1
        prob = prob*valid
        action = np.random.choice(64, p=prob/np.sum(prob))
        if not action in actions:
            # Choose again if prediction is illegal
            #return self.get_action(color, actions)
            return random.choice(actions)
        return action

    # Things to do in one turn
    def turn(self, color):
        actions = self.legal_actions(color)
        if len(actions)>0:
            action = self.get_action(color, actions)
            if color==1:
                tmp = 3*np.ones([8,8], dtype=np.float32)
                state = self.state*(tmp-self.state)*(tmp-self.state)/2
                self.states.append(state)
                self.actions.append(action)
            self.state = self.place_stone(self.state, action, color)
            self.pass_flg = False
            self.stone_num += 1
        else:
            if self.pass_flg:
                self.stone_num = 64 # Game over when two players pass consecutively
            self.pass_flg = True

# Return True if the index is out of the board
def is_outside(pos):
    return pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7
