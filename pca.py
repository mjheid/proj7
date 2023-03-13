import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas  as pd

input_data = np.load("data/input_data.npy")
input_index = np.load("data/input_index.npy", allow_pickle=True)
output_data = np.load("data/output_data.npy")
output_index = np.load("data/output_index.npy", allow_pickle=True)

pca_in = PCA(n_components=input_data.shape[1])
pca_out = PCA(n_components=output_data.shape[1])

input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1]*input_data.shape[2]))
output_data = np.reshape(output_data, (output_data.shape[0], output_data.shape[1]*output_data.shape[2]))

pca_in.fit(input_data)
pca_out.fit(output_data)

pca_input = pca_in.transform(input_data)
pca_output = pca_out.transform(output_data)

input_index = pd.Series(input_index)
output_index = pd.Series(output_index)


fig, ax = plt.subplots()
pops = input_index.unique()
c = 0
for pop in pops:
    data = pca_input[input_index==pop]
    col = np.ones((data.shape[0]))
    col[:] = c
    ax.scatter(data[:,0], data[:,1], c=col, label=pop, cmap='inferno')
    c += 1
ax.legend()
plt.show()