import os
import numpy as np

import chainer
import chainer.links as L
from chainer import training, serializers, cuda, optimizers, Variable
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import SLPolicy

# 1ターンシミュレート
# position  y, x
# color  1:white, 2:black
def turn(state, position, color):
    pos = np.array(position)-[1,1]
    # pos[0]が縦位置、pos[1]が横位置
    state[pos[0], pos[1]] = color

    dys = [-1, -1, -1, 0, 0, 1, 1, 1]
    dxs = [-1, 0, 1, -1, 1, -1, 0, 1]
    for dy,dx in zip(dys, dxs):
        if pos[0]+dy<0 or pos[0]+dy>7 or pos[1]+dx<0 or pos[1]+dx>7:
            continue
        if state[pos[0]+dy, pos[1]+dx]+color!=3:
            continue
        ref = pos + [dy, dx]
        while(state[ref[0], ref[1]]+color==3):
            ref += [dy, dx]
            if ref[0]<0 or ref[0]>7 or ref[1]<0 or ref[1]>7:
                break
        if ref[0]>=0 and ref[0]<=7 and ref[1]>=0 and ref[1]<=7:
            if state[ref[0], ref[1]]==color:
                ref -= [dy, dx]
                while(state[ref[0], ref[1]]+color==3):
                    state[ref[0], ref[1]] = 3-state[ref[0], ref[1]]
                    ref -= [dy, dx]
    return state

def valid_pos(state, color):
    positions = []
    for i in range(8):
        for j in range(8):
            if state[i, j] != 0:
                continue
            # Search 8 directions
            dys = [-1, -1, -1, 0, 0, 1, 1, 1]
            dxs = [-1, 0, 1, -1, 1, -1, 0, 1]
            for dy,dx in zip(dys, dxs):
                if i+dy<0 or i+dy>7 or j+dx<0 or j+dx>7:
                    continue
                if state[i+dy, j+dx]+color!=3:
                    continue
                ref = np.array([i+dy, j+dx])
                while(state[ref[0], ref[1]]+color==3):
                    ref += [dy, dx]
                    if ref[0]<0 or ref[0]>7 or ref[1]<0 or ref[1]>7:
                        break
                if ref[0]>=0 and ref[0]<=7 and ref[1]>=0 and ref[1]<=7:
                    if state[ref[0], ref[1]]==color:
                        positions.append([i+1,j+1])
                        break
    return positions


def show_line(s, i):
    s = str(s).replace(" ", "|").replace("[", "|").replace("]", "|")
    s= s.replace("0", "   ").replace("1", " X ").replace("2", " O ")
    print(str(i+1)+s)

def show(state):
    print("   1   2   3   4   5   6   7   8   ")
    for i in range(8):
        print(" " + "-"*34)
        show_line(state[i], i)
    print(" " + "-"*33)
    print("X(You):"+ str(np.sum(state==1)) + ", O(AI):" + str(np.sum(state==2)) + ", Empty:" + str(np.sum(state==0)))
    print("\n")

def judge(state):
    you = np.sum(state==1)
    ai = np.sum(state==2)
    if you>ai:
        print("You WIN!")
    elif you<ai:
        print("You LOSE")
    else:
        print("DRAW")
    print("X(You):"+ str(np.sum(state==1)) + ", O(AI):" + str(np.sum(state==2)) + ", Empty:" + str(np.sum(state==0)))

def initial_state():
    state = np.zeros([8, 8], dtype=np.int8)
    state[4, 3] = 1
    state[3, 4] = 1
    state[3, 3] = 2
    state[4, 4] = 2
    return state

def safeinput():
    line = input()
    if line.isspace():
        return safeinput()
    line = line.split(',')

    if len(line) != 2:
        print("Try again.")
        return safeinput()
    else:
        return line

def get_input(state, positions):
    position = [int(e) for e in safeinput()]
    pos = np.array(position)-[1,1]
    # pos[0]が縦位置、pos[1]が横位置
    if not position in positions:
        print("This position is invalid. Choose another position")
        return get_input(state, positions)
    else:
        return position


def main():
	print("\n"+"*"*34)
	print("*"*11+"Game Start!!"+"*"*11)
	print("*"*34+"\n")
	model = L.Classifier(SLPolicy.SLPolicyNet(), lossfun=softmax_cross_entropy)
	serializers.load_npz('model.npz', model)
	pass_flg = False
	state = initial_state()
	show(state)
	for play_num in range(60):
	    positions = valid_pos(state, 1)
	    print("Valid choice:", positions)
	    if len(positions)>0:
	        position = get_input(state, positions)
	        state = turn(state, position, 1)
	        show(state)
	        pass_flg = False
	    else:
	        if pass_flg:
	            break
	        print("You pass.")
	        pass_flg = True

	    positions = valid_pos(state, 2)
	    if len(positions)>0:
	        state_var = chainer.Variable(state.reshape(1, 1, 8, 8).astype(np.float32))
	        action_probabilities = model.predictor(state_var).data.reshape(64)
	        idx = np.argmax(action_probabilities)
	        position = [idx//8+1, idx%8+1]
	        state = turn(state, position, 2)
	        show(state)
	        pass_flg = False
	    else:
	        if pass_flg:
	            break
	        print("AI pass")
	        pass_flg = True
	print("\n"+"*"*34)
	print("*"*12+"Game End!!"+"*"*12)
    print("*"*34)
	judge(state)



if __name__ == '__main__':
    main()
