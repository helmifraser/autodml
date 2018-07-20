import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import sys

from coloured_print import printc

parser = argparse.ArgumentParser(description='Plot loss history file')
parser.add_argument('filepath', metavar='filepath', type=str, nargs='+',
                    help='filepath of history file(s)')

args = parser.parse_args()
filepath = args.filepath

if filepath is None:
    printc("Error: filepath == None, no filepath specified, cannot plot data")
    sys.exit(0)

print("Displaying loss plot: {}".format(filepath))

nb_epochs = 20

data = [None] * len(filepath)
# concatenated = [None] * len(filepath)

fig = plt.figure(1, figsize=(12, 8))
fig.suptitle("Training loss")

for idx, path in enumerate(filepath):
    data[idx] = pd.read_csv(path, header=None, names=[
                            i for i in range(nb_epochs)])

    for i in range(nb_epochs):
        if data[idx][i].dtype is np.dtype('object'):
            # print("Found at index ", i)
            data[idx][i] = data[idx][i].str.lstrip('[')
            data[idx][i] = data[idx][i].str.rstrip(']')
            data[idx][i] = pd.to_numeric(data[idx][i])

    # serialised = []
    concatenated = []

    for i in range(len(data[idx].loc[:, 0])):
        episode = []

        for element in data[idx].loc[i, :]:
            if math.isnan(element) is not True:
                episode.append(element)
                concatenated.append(element)

        # serialised.append(episode)

    plt.plot([i for i in range(len(concatenated))], concatenated, label=path[path.rfind("/")+1:])
    plt.legend(loc='upper right')
    plt.xlabel('Epoch (total)')
    plt.ylabel('Loss')
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
