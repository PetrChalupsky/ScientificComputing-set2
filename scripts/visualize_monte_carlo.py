import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


all_grids = np.load(f"data/monte_carlo.npy")
grid = all_grids[-1]
print(all_grids[-1])


cmap = ListedColormap(['white', 'black', 'red'])
plt.imshow(grid, cmap=cmap, interpolation='nearest')

plt.title('Monte carlo cluster')
plt.axis('off')

# black line around figure
ax = plt.gca()
border = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', transform=ax.transAxes)
ax.add_patch(border)

plt.savefig("results/monte_carlo", dpi=300)
plt.show()
