import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


all_grids = np.load(f"data/monte_carlo.npy")
grid = all_grids[-1]
print(all_grids[-1])


# cmap = ListedColormap(['white', 'black', 'red'])
# # cmap = ListedColormap(['black', 'yellow', 'red']) # petr style
# cmap = ListedColormap()
plt.imshow(grid, cmap="inferno", interpolation='nearest')

plt.title('Monte carlo cluster')
plt.axis('off')

# black line around figure
ax = plt.gca()
border = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', transform=ax.transAxes)
ax.add_patch(border)

plt.savefig("results/monte_carlo_final_cluster", dpi=300)


cluster_densities = [np.sum(grid == 1) / all_grids[0].size for grid in all_grids]

plt.figure(figsize=(8, 5))
plt.plot(range(len(all_grids)), cluster_densities, marker='o', linestyle='-')
plt.xlabel("Time Step")
plt.ylabel("Cluster Density")
plt.title("Cluster Density Over Time")
plt.grid(True)
plt.savefig("results/monte_carlo_over_time", dpi=300)

plt.show()
