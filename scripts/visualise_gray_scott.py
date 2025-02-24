"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Visualises final concentrations of Gray-Scott reaction-diffusion model 
    using an explicit finite difference method.
"""

import numpy as np
import matplotlib.pyplot as plt

def visualise_gray_scott(t):
    """
    Visualises final concentration of u for three sets of parameters. 
    """
    f1, k1 = 0.02,0.05
    grid_u1 = np.load("data/gs_0.02_0.05_periodic_5000.npy")
    
    f2, k2 = 0.022, 0.051
    grid_u2 = np.load("data/gs_0.022_0.051_periodic_5000.npy")
    
    f3, k3 = 0.01, 0.033
    grid_u3 = np.load("data/gs_0.01_0.033_periodic_5000.npy")
    
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5), constrained_layout=True)

    im0 = axes[0].imshow(grid_u1, origin="lower", cmap="inferno", vmin=0, vmax=1)
    im1 = axes[1].imshow(grid_u2, origin="lower", cmap="inferno", vmin=0, vmax=1)
    im2 = axes[2].imshow(grid_u3, origin="lower", cmap="inferno", vmin=0, vmax=1)
    cbar = fig.colorbar(im0, ax=axes[:], fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=14)  # Set colorbar tick label size

    axes[0].set_ylabel("$y$", fontsize=18)
    axes[1].set_xlabel("$x$", fontsize=18)
    axes[0].set_title("$f$=%1.3f," %f1 + "$k$=%1.3f" %k1, fontsize=18)
    axes[1].set_title("$f$=%1.3f," %f2 + "$k$=%1.3f" %k2, fontsize=18)
    axes[2].set_title("$f$=%1.3f," %f3 + "$k$=%1.3f" %k3, fontsize=18)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=16)
        
    fig.suptitle("Concentration of $u$, $t$=%i" %t, fontsize=16)
    
    return fig

def visualise_gray_scott_noise(t):
    """
    Visualises final concentration of u for three sets of parameters
    when noise is added to the initial state. 
    """

    f1, k1 = 0.02,0.05
    grid_u1 = np.load("data/gs_noise_0.02_0.05_periodic_5000.npy")
    
    f2, k2 = 0.022, 0.051
    grid_u2 = np.load("data/gs_noise_0.022_0.051_periodic_5000.npy")
    
    f3, k3 = 0.01, 0.033
    grid_u3 = np.load("data/gs_noise_0.01_0.033_periodic_5000.npy")
    
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5), constrained_layout=True)

    im0 = axes[0].imshow(grid_u1, origin="lower", cmap="inferno", vmin=0, vmax=1)
    im1 = axes[1].imshow(grid_u2, origin="lower", cmap="inferno", vmin=0, vmax=1)
    im2 = axes[2].imshow(grid_u3, origin="lower", cmap="inferno", vmin=0, vmax=1)
    cbar = fig.colorbar(im0, ax=axes[:], fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=14)  # Set colorbar tick label size

    axes[0].set_ylabel("$y$", fontsize=18)
    axes[1].set_xlabel("$x$", fontsize=18)
    axes[0].set_title("$f$=%1.3f," %f1 + "$k$=%1.3f" %k1, fontsize=18)
    axes[1].set_title("$f$=%1.3f," %f2 + "$k$=%1.3f" %k2, fontsize=18)
    axes[2].set_title("$f$=%1.3f," %f3 + "$k$=%1.3f" %k3, fontsize=18)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=16)
        
    fig.suptitle("Concentration of $u$, $t$=%i" %t, fontsize=16)
    
    return fig

t = 5000

fig1 = visualise_gray_scott(t)
fig2 = visualise_gray_scott_noise(t)

fig1.savefig("results/gs_5000.png", dpi=300, bbox_inches="tight")
fig2.savefig("results/gs_noise_5000.png", dpi=300, bbox_inches="tight")