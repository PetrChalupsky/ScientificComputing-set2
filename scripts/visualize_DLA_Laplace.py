import matplotlib.pyplot as plt
import numpy as np

cluster = []
diffusion = []
eta_list = [1, 0.2, 1.5]

fig2, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
plt.subplots_adjust(wspace=-0.01, hspace=0.25)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

for eta in eta_list: 
    cluster.append(np.load(f'data/dla_cluster_eta{eta}.npy'))
    diffusion.append(np.load(f'data/dla_diffusion_eta{eta}.npy'))


for i, eta in enumerate(eta_list):
    axes[0, i].imshow(
        cluster[i], origin="lower", cmap="inferno", extent=[0, 1, 0, 1]
    )
    axes[0, i].set_title("eta=" + str(eta))

for i, eta in enumerate(eta_list):
    axes[1, i].imshow(
        diffusion[i], origin="lower", cmap="inferno", extent=[0, 1, 0, 1]
    )

#fig2.colorbar(im, ax=axes[:, :], fraction=0.05, pad=0.025)
fig2.supxlabel("$x$-coordinate", fontsize=12)
fig2.supylabel("$y$-coordinate", fontsize=12, x=0.1)

plt.show()
