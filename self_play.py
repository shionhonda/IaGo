import numpy as np
import chainer
from chainer import Variable
from game import Game

class SelfGame(Game):
    # Get position to place stone
    def get_position_self(self, color, positions):
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

        action_probabilities -= np.min(action_probabilities) # Add bias to make all components non-negative
        idx = np.random.choice(64, p=action_probabilities/np.sum(action_probabilities))
        position = [idx//8+1, idx%8+1]
        if not position in positions:
            # Choose again if prediction is illegal
            return self.get_position_self(color, positions)
            # position = random.choice(positions)
        return position

    def show_self(self):
        print("   1   2   3   4   5   6   7   8   ")
        for i in range(8):
            print(" " + "-"*34)
            s = str(self.state[i]).replace(" ", "|").replace("[", "|").replace("]", "|")
            s= s.replace("0", "   ").replace("1", " X ").replace("2", " O ").replace(".", "")
            print(str(i+1)+s)
        print(" " + "-"*33)
        print("X(AI1):"+ str(np.sum(self.state==1)) + ", O(AI2):" + str(np.sum(self.state==2))\
        + ", Empty:" + str(np.sum(self.state==0)))
        print("\n")

    # Things to do in one turn
    def turn_self(self, color):
        players = ["AI1", "AI2"]
        positions = self.valid_pos(color)
        print("Valid choice:", positions)
        if len(positions)>0:
            position = self.get_position_self(color, positions)
            self.place_stone(position, color)
            self.show_self()
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

    # Judge game winner
    def judge_self(self):
        ai1 = np.sum(self.state==1)
        ai2 = np.sum(self.state==2)
        if ai1>ai2:
            print("AI1 WIN!")
        elif ai1<ai2:
            print("AI1 LOSE")
        else:
            print("DRAW")
        return "X(AI1):"+ str(ai1) + ", O(AI2):" + str(ai2) + ", Empty:" + str(np.sum(self.state==0))



# Whole game
def main():
    print("\n"+"*"*34)
    print("*"*11+"Game Start!!"+"*"*11)
    print("*"*34+"\n")
    game = SelfGame()
    game.show_self()

    while(game.stone_num<64):
        game.turn_self(1)
        game.turn_self(2)

    print("\n"+"*"*34)
    print("*"*12+"Game End!!"+"*"*12)
    print("*"*34)
    jd = game.judge_self()
    print(jd)
    game.gamelog += jd + "\n"
    game.save_gamelog()

if __name__ == '__main__':
    main()
