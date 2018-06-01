import os
from datetime import datetime
import numpy as np
import random
import chainer
import chainer.links as L
from chainer import training, serializers, cuda, optimizers, Variable
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
import SLPolicy

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
        self.play_num = 1
        self.pass_flg = False
        self.date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.gamelog = "IaGo v1.1\n" + self.date + "\n"
        # Load model
        self.model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
        serializers.load_npz('model.npz', self.model)
        self.model2 = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
        serializers.load_npz('model.npz', self.model2)

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

    # Convert ndarray to a board-like string
    def show(self):
        print("   1   2   3   4   5   6   7   8   ")
        for i in range(8):
            print(" " + "-"*34)
            s = str(self.state[i]).replace(" ", "|").replace("[", "|").replace("]", "|")
            s= s.replace("0", "   ").replace("1", " X ").replace("2", " O ").replace(".", "")
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
            print("You WIN!")
        elif you<ai:
            print("You LOSE")
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
    def get_position(self, color, positions):
        # Your turn
        if color==1:
            position = [int(e) for e in self.safeinput()]
            if not position in positions:
                print("This position is invalid. Choose another position")
                return self.get_position(color, positions) # Recurse
        # AI's turn
        else:
            # Predict position to place stone
            X = np.stack([self.state==1, self.state==2], axis=2)
            state_var = chainer.Variable(X.reshape(1, 2, 8, 8).astype(np.float32))
            action_probabilities = self.model.predictor(state_var).data.reshape(64)
            idx = np.argmax(action_probabilities)
            position = [idx//8+1, idx%8+1]
            if not position in positions:
                # Choose randomly if prediction is illegal (very rare)
                position = random.choice(positions)
        return position

    # Things to do in one turn
    def turn(self, color):
        players = ["You", "AI"]
        positions = self.valid_pos(color)
        print("Valid choice:", positions)
        if len(positions)>0:
            position = self.get_position(color, positions)
            self.place_stone(position, color)
            self.show()
            self.pass_flg = False
            self.gamelog += "[" + str(self.play_num) + "]" + players[color-1]\
             + ": " + str(position) + "\n"
            self.stone_num += 1
        else:
            if self.pass_flg:
                self.stone_num = 64 # Game over when two players pass consecutively
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

# Whole game
def main():
    print("\n"+"*"*34)
    print("*"*11+"Game Start!!"+"*"*11)
    print("*"*34+"\n")
    game = Game()
    game.show()

    while(game.stone_num<64):
        game.turn(1)
        game.turn(2)

    print("\n"+"*"*34)
    print("*"*12+"Game End!!"+"*"*12)
    print("*"*34)
    jd = game.judge()
    print(jd)
    game.gamelog += jd + "\n"
    game.save_gamelog()

if __name__ == '__main__':
    main()
