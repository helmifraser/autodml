import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import sys

from coloured_print import printc

parser = argparse.ArgumentParser(description='Plot loss history file')
parser.add_argument('val_path', metavar='val_path', type=str,
                    help='filepath of validation score file')

# parser.add_argument('hist_path', metavar='hist_path', type=str, nargs='1',
#                     help='filepath of history file')

args = parser.parse_args()
val_path = args.val_path
# hist_path = args.hist_path

if val_path is None:
    printc("Error: val_path == None, no val_path specified, cannot plot data")
    sys.exit(0)
# elif hist_path is None:
#     printc("Error: hist_path == None, no hist_path specified, cannot plot data")
#     sys.exit(0)


printc("Displaying val plot: {}".format(val_path), 'warn')

fig = plt.figure(1, figsize=(20, 8))
fig.suptitle("Validation loss per episode in validation set")

data = pd.read_csv(val_path, header=None, names=[i for i in range(5)])
objects = [str(i) for i in range(data.shape[0])]
x = np.arange(len(objects))

plt.subplot(121)
for i in range(5):
    if data[i].dtype is np.dtype('object'):
        # print("Found at index ", i)
        data[i] = data[i].str.lstrip('[')
        data[i] = data[i].str.rstrip(']')
        data[i] = pd.to_numeric(data[i])

mean_val = sum(data[0])/len(data[0])
plt.axhline(mean_val, color='r', linestyle='--')
plt.bar(x, data[0], width=0.8,
        align='center', ec="black", tick_label=objects)
# plt.xticks(x_pos, objects)
plt.xlabel('Episode')
plt.ylabel('Loss')

# path = val_path[:val_path.rfind("/")+1] + "history_file.txt"
# data = pd.read_csv(path, header=None, names=[i for i in range(5)])

# if data[0].dtype is np.dtype('object'):
#     # print("Found at index ", i)
#     data[0] = data[0].str.lstrip('[')
#     data[0] = data[0].str.rstrip(']')
#     data[0] = pd.to_numeric(data[0])
# print(data)

# plt.subplot(122)
#
# plt.bar(x, data[0], align='center', alpha=0.5)
# plt.xticks(x, objects)
# plt.xlabel('Episode')
# plt.ylabel('Loss')


plt.grid(True)
plt.show()
