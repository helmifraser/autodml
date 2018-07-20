import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser(description='Plot loss history file')
parser.add_argument('filepath', metavar='filepath', type=str,
                    help='filepath of history file')

args = parser.parse_args()
filepath = args.filepath
print("Displaying loss plot: {}".format(filepath))

nb_epochs = 20

data = pd.read_csv(filepath, header=None,
                   names=[i for i in range(nb_epochs)], prefix="Epoch_")

for i in range(nb_epochs):
    if data[i].dtype is np.dtype('object'):
        # print("Found at index ", i)
        data[i] = data[i].str.lstrip('[')
        data[i] = data[i].str.rstrip(']')
        data[i] = pd.to_numeric(data[i])

serialised = []
concatenated = []

for i in range(len(data.loc[:, 0])):
    episode = []

    for element in data.loc[i, :]:
        if math.isnan(element) is not True:
            episode.append(element)
            concatenated.append(element)

    serialised.append(episode)


fig = plt.figure(1, figsize=(12, 8))
fig.suptitle("Training loss")

plt.plot([i for i in range(len(concatenated))], concatenated)
plt.grid(True)

# count = len(serialised[0][:])
# for i in range(np.shape(serialised)[0]):
#     if i > 0:
#         count = count + len(serialised[i-1][:])
#         plt.plot([j for j in range(count, count + len(serialised[i][:]))], serialised[i][:])
#     else:
#         plt.plot([j for j in range(0, count)], serialised[i][:])

# for i in range(0, nb_epochs):
#     plt.subplot(4, 5, i+1)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Episode '+ str(i+1))
#     data.loc[i].plot()
#     plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
