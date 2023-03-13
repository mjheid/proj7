import numpy as np
import pandas as pd


def random_sample(samples, n):
    rng = np.random.default_rng()
    return rng.choice(samples, n)


def random_parents(num_of_children, origin_parents, data):
    ordered_pop = origin_parents.value_counts(ascending=True)
    rng = np.random.default_rng()

    children_origin = []
    children_parents = np.zeros((num_of_children, 2, data.shape[1], 2))
    children_parents[:,:] = -1

    created_children = 0
    pop = ordered_pop.index[0]

    while created_children < num_of_children:
        children_origin.append(pop)

        pop_index = np.where(origins==pop)[0][0]
        parents = rng.choice(origins, 2, p=geneflow[pop_index])
        children_parents[created_children,0,:,:] = data[rng.choice(origin_parents[origin_parents == parents[0]].index, 1)[0]]
        children_parents[created_children,1,:] = data[rng.choice(origin_parents[origin_parents == parents[1]].index, 1)[0]]

        created_children += 1
        pop = ordered_pop.index[created_children%(ordered_pop.shape[0])]

    return children_parents, children_origin


def create_offspring(childrens_parents):
    rng = np.random.default_rng()
    
    children_arr  = np.zeros((childrens_parents.shape[0], childrens_parents.shape[2], childrens_parents.shape[3]))
    children_arr[:,:,:] = -1

    children = rng.choice(childrens_parents, 1, axis=3)
    children_arr[:,:,0] = children[:,0,:,0]
    children_arr[:,:,1] = children[:,1,:,0]

    return children_arr


p_dist = pd.read_csv("data/p_distances.csv")
data = pd.read_csv("data/data.ped", header=None, usecols=range(16), sep="\t")

kek=data.copy()
populations = data[0].to_numpy()
data = data.drop(columns=[0,1,2,3,4,5])
data.columns = list(range(data.columns.shape[0]))
data = [np.array([list(map(int, row.split())) for row in data[col]]) for col in data.columns]
data = np.stack(data, axis=0)
data = data.transpose((1,0,2))

np.save("data/input_data", data)
np.save("data/input_index", populations)

origins = p_dist.country1.unique()
geneflow = np.reshape(p_dist.p_dist.to_numpy(), (origins.shape[0], origins.shape[0]))

N = 1000
time_steps = 100
mortality = 0.5

if data.shape[0] < N:
    new_data = np.zeros((N, data.shape[1], data.shape[2]))
    new_populations = pd.Series(["0"]*N)

    new_data[:data.shape[0]] = data
    new_populations[:data.shape[0]] = populations

    n_comb = new_data.shape[0] - data.shape[0]
    comb_m, children_origin = random_parents(n_comb, new_populations[:data.shape[0]], new_data[:data.shape[0]])
    new_populations[data.shape[0]:] = children_origin
    new_data[data.shape[0]:] = create_offspring(comb_m)

    data = new_data
    populations = new_populations

for ts in list(range(time_steps)):
    new_data = np.zeros(data.shape)
    new_data[:,:,:] = -1
    new_populations = pd.Series(["0"]*N)

    
    start_pos = 0
    for pop in origins:
        end_pos = int(mortality*populations[populations==pop].shape[0]+start_pos)
        new_data[start_pos:end_pos] = random_sample(data[populations==pop], end_pos-start_pos)
        new_populations[start_pos:end_pos] = pop
        start_pos = end_pos
    
    n_comb = new_data.shape[0] - start_pos
    comb_m, children_origin = random_parents(n_comb, new_populations[:start_pos-1], new_data[:start_pos-1])
    new_populations[start_pos:] = children_origin
    new_data[start_pos:] = create_offspring(comb_m)

    data = new_data
    populations = new_populations

np.save("data/output_data", data)
np.save("data/output_index", populations)
