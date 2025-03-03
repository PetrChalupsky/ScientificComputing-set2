"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Visualize DLA from Laplace equation.
"""
import matplotlib.pyplot as plt
import numpy as np

cluster = []
diffusion = []
eta_list = [0.2, 1, 1.5]

fig2, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True, sharex=True, layout='constrained')

for eta in eta_list: 
    cluster.append(np.load(f'data/dla_cluster_eta{eta}.npy'))
    diffusion.append(np.load(f'data/dla_diffusion_eta{eta}.npy'))


for i, eta in enumerate(eta_list):
    axes[0, i].imshow(
        cluster[i], origin="lower", cmap="inferno", extent=[0, 1, 0, 1]
    )
    axes[0, i].set_title("$\eta$ =" + str(eta))

for i, eta in enumerate(eta_list):
    im = axes[1, i].imshow(
        diffusion[i], origin="lower", cmap="inferno", extent=[0, 1, 0, 1]
    )

cbar = fig2.colorbar(im, ax=axes[:,:], pad = 0.01)
fig2.supxlabel("$x$-coordinate", fontsize=12, x=0.5)
fig2.supylabel("$y$-coordinate", fontsize=12)

plt.savefig('results/DLA_Laplace_heatmaps', dpi=300)
