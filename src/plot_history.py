import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nb_epochs = 20

data = pd.read_csv("../misc/first_history.txt", header=None,
                   names=[i for i in range(nb_epochs)], prefix="Epoch_")


for i in range(nb_epochs):
    if data[i].dtype is np.dtype('object'):
        # print("Found at index ", i)
        data[i] = data[i].str.lstrip('[')
        data[i] = data[i].str.rstrip(']')
        data[i] = pd.to_numeric(data[i])

# print(data.loc[1])

fig = plt.figure(1, figsize=(12, 8))
fig.suptitle("Training loss per episode")

for i in range(0, nb_epochs):
    plt.subplot(4, 5, i+1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Episode '+ str(i+1))
    data.loc[i].plot()
    plt.grid(True)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
