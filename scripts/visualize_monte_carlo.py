import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


final_cluster_1 =  np.load(f"data/monte_carlo_final_cluster_0.2.npy")
final_cluster_2 =  np.load(f"data/monte_carlo_final_cluster_0.4.npy")
final_cluster_3 =  np.load(f"data/monte_carlo_final_cluster_0.6.npy")
final_cluster_4 =  np.load(f"data/monte_carlo_final_cluster_0.8.npy")
final_cluster_5 =  np.load(f"data/monte_carlo_final_cluster_1.0.npy")

final_clusters = [final_cluster_1, final_cluster_2, final_cluster_3, final_cluster_4, final_cluster_5]
p_values = [0.2, 0.4, 0.6, 0.8, 1.0]

# plot results final clusters for varying p values
for i, elem in enumerate(final_clusters):
    cmap = ListedColormap(['black', 'pink', 'black'])
    plt.imshow(elem, cmap=cmap, interpolation='nearest')
    plt.title(f'Monte Carlo final cluster {p_values[i]}')
    plt.axis('off')

    # Add black border
    ax = plt.gca()
    border = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', transform=ax.transAxes)
    ax.add_patch(border)

    plt.savefig(f"results/monte_carlo_final_cluster_{i}.png", dpi=300)


all_grids_1 =  np.load(f"data/monte_carlo_all_grids_0.2.npy")
all_grids_2 =  np.load(f"data/monte_carlo_all_grids_0.4.npy")
all_grids_3 =  np.load(f"data/monte_carlo_all_grids_0.6.npy")
all_grids_4 =  np.load(f"data/monte_carlo_all_grids_0.8.npy")
all_grids_5 =  np.load(f"data/monte_carlo_all_grids_1.0.npy")

all_grids = [all_grids_1, all_grids_2, all_grids_3, all_grids_4, all_grids_5]

for i, all_grids in enumerate(all_grids):
    cluster_densities = [np.sum(grid == 1) / all_grids[0].size for grid in all_grids]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(all_grids)), cluster_densities, marker='o', linestyle='-')
    plt.xlabel("Time Step")
    plt.ylabel("Cluster Density")
    plt.title("Cluster Density Over Time")
    plt.grid(True)
    plt.savefig(f"results/monte_carlo_over_time_{i}", dpi=300)

plt.show()
