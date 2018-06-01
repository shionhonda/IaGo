# At least 25GB RAM is required to run this code

import numpy as np

# Transform string data into numpy state and action
def transform(string):
    flat = string.replace("\n", "").split(" ")
    state = np.array([int(flat[j]) for j in range(64)]).reshape(8,8)
    action = (int(flat[65])-1)*8 + int(flat[64])-1
    return state, action

# Rotate indices to 90 degrees (counterclockwise)
def rotate(action):
    y, x = action//8-3.5, action%8-3.5
    y_, x_ = -x+3.5, y+3.5
    return y_*8+x_

# Transpose indices
def transpose(action):
    y, x = action//8-3.5, action%8-3.5
    y_, x_ = x+3.5, y+3.5
    return y_*8+x_

def main():
    with open("data/data.txt", "r") as f:
        data = f.readlines()

    # Plays by Black are to be translated into plays by White
    B_data = []
    W_data = []
    for line in data:
        if 'B' in line:
            B_data.append(line)
        elif 'W' in line:
            W_data.append(line)
    len_B = len(B_data)
    len_W = len(W_data)

    # List -> ndarray
    states = np.zeros([len_B+len_W, 8, 8])
    actions = np.zeros(len_B+len_W)
    for i in range(0, len_B):
        states[i,:,:], actions[i] = transform(B_data[i])
    for i in range(len_B, len_B+len_W):
        st, actions[i] = transform(W_data[i-len_B])
        st[np.where(st==0)] = 3
        states[i,:,:] = 3-st
    del B_data, W_data # Memory release

    # Data augmentation
    S = states
    A = actions
    # Rotate
    for i in range(3):
        states = np.rot90(states, k=1, axes=(1,2))
        S = np.concatenate([S, states], axis=0)
        actions = rotate(actions)
        A = np.concatenate([A, actions], axis=0)
    # Transpose
    states = states.transpose(0,2,1)
    S = np.concatenate([S, states], axis=0)
    actions = transpose(actions)
    A = np.concatenate([A, actions], axis=0)
    # Rotate
    for i in range(3):
        states = np.rot90(states, k=1, axes=(1,2))
        S = np.concatenate([S, states], axis=0)
        actions = rotate(actions)
        A = np.concatenate([A, actions], axis=0)
    del states, actions

    # Save data
    np.save('data/states.npy', S)
    np.save('data/actions.npy', A)
    del S, A

if __name__ == '__main__':
    main()
