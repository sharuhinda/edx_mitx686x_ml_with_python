import numpy as np
import kmeans
import common
import naive_em
import em

import matplotlib.pyplot as plt

from common import GaussianMixture

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
"""
The task was: Try K=range(1, 5) on this data, plotting each solution using our common.plot function.
Since the initialization is random, please use seeds=range(5) to and select the one that minimizes the total cost.
Save the associated plots (best solution for each K).

First I thought that I have to evaluate the cost function right after initialization and do it by myself. So I created
function get_cost_kmeans to calculate the cost right after initialization. But then I have read the helpful post
in discussion forum where it was explaiend that cost evaluation should be done after algorthm have converged. So the 
function remained unused
"""
def get_cost_kmeans(X, mixture: GaussianMixture):
    """
    Returns cost as sum of minimal distances to cluster centers
    """
    n = X.shape[0]
    K = mixture.mu.shape[0]
    cost = 0
    for i in range(n):
        tiled_vector = np.tile(X[i, :], (K, 1))
        sse = ((tiled_vector - mixture.mu)**2).sum(axis=1)
        cost += np.min(sse)
    return cost

ks = list(range(1,5))
seeds = list(range(5))

fig, axs = plt.subplots(len(seeds), len(ks), figsize=(5*len(ks), 4*len(seeds)))
min_costs = {k: None for k in ks}
min_seeds = {k: None for k in ks}
for k in ks:
    for seed in seeds:
        mixture, post = common.init(X=X, K=k, seed=seed)
        mixture, post, cost = kmeans.run(X, mixture=mixture, post=post)
        if min_costs[k] is None:
            min_costs[k] = cost
            min_seeds[k] = seed
        elif cost < min_costs[k]:
            min_costs[k] = cost
            min_seeds[k] = seed
        common.plot(X=X, mixture=mixture, post=post, title=f'Cost: {cost:.4f}', ax=axs[seed, k-1])
plt.show()