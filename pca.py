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


fig, ax = plt.subplots(figsize=(12, 6))
pops = input_index.unique()
norm = plt.Normalize(vmin=0, vmax=len(pops)) # create a normalization function for colormap
cmap = plt.cm.jet # choose a colormap
for i, pop in enumerate(pops):
    data = pca_input[input_index==pop]
    ax.scatter(data[:,0], data[:,1], c=cmap(norm(i)), label=pop)

# set legend properties to show all labels and move it outside of the plot
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, ncol=2, fontsize='small', title='Populations', title_fontsize='medium',
                   bbox_to_anchor=(1.05, 1), loc='upper left')
# adjust the spacing between the plot and the legend
plt.subplots_adjust(right=0.75)

ax.set_ylabel("PCA2")
ax.set_xlabel("PCA1")

plt.savefig("data/input_pca.png")#
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
pops = input_index.unique()
norm = plt.Normalize(vmin=0, vmax=len(pops)) # create a normalization function for colormap
cmap = plt.cm.jet # choose a colormap
for i, pop in enumerate(pops):
    data = pca_output[output_index==pop]
    ax.scatter(data[:,0], data[:,1], c=cmap(norm(i)), label=pop)

# set legend properties to show all labels and move it outside of the plot
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, ncol=2, fontsize='small', title='Populations', title_fontsize='medium',
                   bbox_to_anchor=(1.05, 1), loc='upper left')
# adjust the spacing between the plot and the legend
plt.subplots_adjust(right=0.75)

ax.set_ylabel("PCA2")
ax.set_xlabel("PCA1")

plt.savefig("data/output_pca.png")#
plt.close()