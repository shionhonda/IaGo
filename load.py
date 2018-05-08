import numpy as np

with open("data/data.txt", "r") as f:
    data = f.readlines()

# Use plays by Black in games where Black wins
B_data = []
for line in data:
    if 'B' in line:
        B_data.append(line)
print(B_data[0:5])
leng = len(B_data)

# List -> ndarray 
states = np.zeros([leng, 8, 8])
actions = np.zeros(leng)
for i in range(leng):
    flat = B_data[i]
    flat = flat.replace("\n", "").split(" ")
    states[i,:,:] = np.array([int(flat[j]) for j in range(64)]).reshape(8,8)
    actions[i] = (int(flat[65])-1)*8 + int(flat[64])-1

# Save data
np.save('data/states.npy', states)
np.save('data/actions.npy', actions)