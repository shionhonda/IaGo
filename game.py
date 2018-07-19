import os
import argparse
from datetime import datetime
import numpy as np
from numba import jit
from chainer import Variable, serializers
import policy
import MCTS
#from game_func import GameFunctions as GameFunctions

class Game:

    def __init__(self, auto):
        # Initialize board state
        if auto:
            self.p1 = "IaGo(SLPolicy)"
            self.model = policy.SLPolicy()
            serializers.load_npz('./models/sl_model.npz', self.model)
        else:
            self.p1 = "You"
            self.model = None
        self.p2 = "IaGo(APV-MCTS)"

        self.state = np.zeros([8, 8], dtype=np.float32)
        self.state[4, 3] = 1
        self.state[3, 4] = 1
        self.state[3, 3] = 2
        self.state[4, 4] = 2
        # Initialize game variables
        self.stone_num = 4
        self.play_num = 1
        self.pass_flg = False
        self.date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.gamelog = "IaGo v2.0\n" + self.date + "\n"
        self.mcts = MCTS.MCTS()

    # Convert ndarray to a board-like string
    def show(self):
        print("   1   2   3   4   5   6   7   8   ")
        for i in range(8):
            print(" " + "-"*33)
            s = str(self.state[i]).replace(" ", "").replace(".", "|").replace("[", "|").replace("]", "")
            s= s.replace("0", "   ").replace("1", " X ").replace("2", " O ")
            print(str(i+1)+s)
        print(" " + "-"*33)
        print("X(You):"+ str(np.sum(self.state==1)) + ", O(AI):" + str(np.sum(self.state==2))\
        + ", Empty:" + str(np.sum(self.state==0)))
        print("\n")

    # Judge game winner
    def judge(self):
        you = np.sum(self.state==1)
        ai = np.sum(self.state==2)
        if you>ai:
            print(self.p1, "WIN!")
        elif you<ai:
            print(self.p1, "LOSE")
        else:
            print("DRAW")
        return "X(You):"+ str(you) + ", O(AI):" + str(ai) + ", Empty:" + str(np.sum(self.state==0))

    # Input function tolerant for mistyping
    def safeinput(self):
        line = input()
        if line.isspace():
            return safeinput() # Recurse this function for mistyping
        line = line.split(',')
        if len(line) != 2:
            print("Try again.")
            return safeinput() # Recurse this function for mistyping
        else:
            return line

    # Get position to place stone
    def get_action(self, color, actions):
        # Player1's turn
        if color==1:
            position = [int(e) for e in self.safeinput()]
            action = (position[0]-1)*8 + (position[1]-1)
            if not action in actions:
                print("This position is invalid. Choose another position")
                return self.get_action(color, actions) # Recurse
            #action = (position[0]-1)*8 + (position[1]-1)
        # Player2's turn
        else:
            # Choose position by MCTS
            state_var = GameFunctions.make_state_var(self.state, 2)
            action = self.mcts.get_move(self.state, 2)
            self.mcts.update_with_move(action)
        return action

    # Get position to place stone
    def get_action_auto(self, color, actions):
        state_var = GameFunctions.make_state_var(self.state, color)
        # Player1's turn
        if color==1:
            prob = self.model(state_var).data.reshape(64)
            valid = np.zeros(64)
            valid[actions] = 1
            prob = prob*valid
            action = np.random.choice(64, p=prob/np.sum(prob))
            if not action in actions:
                # Choose again if prediction is illegal
                return self.get_action_auto(color, actions)
        # Player2's turn
        else:
            # Choose position by MCTS
            action = self.mcts.get_move(self.state, 2)
            self.mcts.update_with_move(action)
        return action

    # Things to do in one turn
    def turn(self, color, auto):
        players = [self.p1, self.p2]
        actions = GameFunctions.legal_actions(self.state, color)
        print("Valid choice:", GameFunctions.ac2pos(actions))
        if len(actions)>0:
            if auto:
                action = self.get_action_auto(color, actions)
            else:
                action = self.get_action(color, actions)
            self.state = GameFunctions.place_stone(self.state, action, color)
            position = [action//8+1, action%8+1]
            print(position)
            self.show()
            self.pass_flg = False
            self.gamelog += "[" + str(self.play_num) + "]" + players[color-1]\
             + ": " + str(position) + "\n"
            self.stone_num += 1
            print(self.stone_num, "stones")
        else:
            if self.pass_flg:
                self.stone_num = 64 # Game over when two players pass consecutively
                print("2 consecutive passes")
            print(players[color-1] + " pass.")
            self.pass_flg = True
            self.gamelog += "[" + str(self.play_num) + "]" + players[color-1] + ": Pass\n"
        self.play_num += 1

    # Save gamelog
    def save_gamelog(self):
        filename = "./gamelog/"+self.date+".txt"
        file_path = os.path.dirname(filename)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(filename, 'w') as f:
            f.write(self.gamelog)

class GameFunctions:

    @classmethod
    def ac2pos(self, actions):
        positions = []
        for action in actions:
            positions.append([action//8+1, action%8+1])
        return positions

    # Return True if the index is out of the board
    @classmethod
    def is_outside(self, pos):
        return pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7

    @classmethod
    def make_state_var(self, state, color):
        if color==1:
            tmp = 3*np.ones([8,8], dtype=np.float32)
            state = state*(tmp-state)*(tmp-state)/2
        state_2ch = np.stack([state==1, state==2], axis=0).astype(np.float32)
        state_var = Variable(state_2ch.reshape(2,1,8,8).transpose(1,0,2,3))
        return state_var

    # Place a stone and turn all the sandwithced stones
    # Position y:vertical, x:horizontal
    # Color  1:white, 2:black
    @classmethod
    def place_stone(self, state, action, color):
        # Place the stone
        pos = np.array([action//8, action%8])
        state[pos[0], pos[1]] = color
        # Search for sandwitched stones
        dys = [-1, -1, -1, 0, 0, 1, 1, 1] # Search direction
        dxs = [-1, 0, 1, -1, 1, -1, 0, 1] # Search direction
        for dy,dx in zip(dys, dxs):
            if self.is_outside(pos+[dy,dx]):
                continue # Search next direction if index goes out of range
            if state[pos[0]+dy, pos[1]+dx]+color!=3:
                continue # Search next direction if empty or same-color stone
            ref = pos + [dy, dx] # Decide direction
            while(state[ref[0], ref[1]]+color==3):
                ref += [dy, dx] # Referring index
                if self.is_outside(ref):
                    break # Stop if referring index goes out of range
            if self.is_outside(ref):
                continue # Search next direction if index goes out of range
            # Turn sandwitched stones
            if state[ref[0], ref[1]]==color:
                ref -= [dy, dx]
                while(state[ref[0], ref[1]]+color==3):
                    state[ref[0], ref[1]] = 3-state[ref[0], ref[1]]
                    ref -= [dy, dx]
        return state

    @classmethod
    def legal_actions(self, state, color):
        actions = []
        for i in range(8):
            for j in range(8):
                if state[i, j] != 0:
                    continue
                # Search 8 directions
                dys = [-1, -1, -1, 0, 0, 1, 1, 1]
                dxs = [-1, 0, 1, -1, 1, -1, 0, 1]
                for dy,dx in zip(dys, dxs):
                    if self.is_outside([i+dy, j+dx]):
                        continue
                    if state[i+dy, j+dx]+color!=3:
                        continue
                    ref = np.array([i+dy, j+dx])
                    while(state[ref[0], ref[1]]+color==3):
                        ref += [dy, dx]
                        out_flg = self.is_outside(ref)
                        if out_flg:
                            break # Stop if referring index goes out of range
                    if out_flg:
                        continue
                    if state[ref[0], ref[1]]==color:
                        actions.append(i*8+j)
                        break
        return actions




# Whole game
def main():
    parser = argparse.ArgumentParser(description='IaGo:')
    parser.add_argument('--auto', '-a', type=bool, default=False, help='Set True for auto play between MCTS and SLPolicy')
    args = parser.parse_args()

    print("\n"+"*"*34)
    print("*"*11+"Game Start!!"+"*"*11)
    print("*"*34+"\n")

    game = Game(args.auto)
    game.show()

    while(game.stone_num<64):
        game.turn(1, args.auto)
        game.turn(2, args.auto)

    print("\n"+"*"*34)
    print("*"*12+"Game End!!"+"*"*12)
    print("*"*34)
    jd = game.judge()
    print(jd)
    game.gamelog += jd + "\n"
    game.save_gamelog()

if __name__ == '__main__':
    main()
