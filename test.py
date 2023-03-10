import numpy as np
import pandas as pd


def random_sample(samples, n):
    rng = np.random.default_rng()
    return rng.choice(samples, n)


def random_parents(num_possible_parents, num_of_children, origin_parents):
    return 0, 0


def create_offspring(individuals, childrens_parents):
    return 0


p_dist = pd.read_csv("data/p_distances.csv")
data = pd.read_csv("data/MARITIME_ROUTE.ped", header=None, usecols=range(1006), sep="\t")

kek=data.copy()
populations = data[0].to_numpy()
data = data.drop(columns=[0,1,2,3,4,5])
data.columns = list(range(data.columns.shape[0]))
data = [np.array([list(map(int, row.split())) for row in data[col]]) for col in data.columns]
data = np.stack(data, axis=0)
data = data.transpose((1,0,2))

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
    comb_m, children_origin = random_parents(data.shape[0], n_comb, new_populations[:data.shape[0]])
    new_populations[data.shape[0]+1:] = children_origin
    new_data[data.shape[0]+1:] = create_offspring(new_data[:data.shape[0]], comb_m)

    data = new_data
    populations = new_populations

for ts in list(range(time_steps)):
    new_data = np.zeros(data.shape)
    new_populations = populations.copy()
    
    start_pos = 0
    for pop in origins:
        end_pos = int(mortality*populations[populations==pop].shape[0]+start_pos)
        new_data[start_pos:end_pos] = random_sample(data[pop==pop[0]], end_pos-start_pos)
        new_populations[start_pos:end_pos] = pop
        start_pos = end_pos + 1
    
    n_comb = new_data.shape[0] - start_pos -1
    comb_m, children_origin = random_parents(start_pos-1, n_comb, new_populations[:start_pos-1])
    new_populations[start_pos:] = children_origin
    new_data[start_pos:] = create_offspring(new_data[:start_pos-1], comb_m)

    data = new_data
    populations = new_populations
