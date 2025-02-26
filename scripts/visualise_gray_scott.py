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
    
    f3, k3 = 0.035, 0.060
    grid_u3 = np.load("data/gs_0.035_0.060_periodic_5000.npy")
    
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
    
    f3, k3 = 0.035, 0.060
    grid_u3 = np.load("data/gs_noise_0.035_0.060_periodic_5000.npy")
    
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

def visualise_concentrations(t):
    """
    Visualises the sum of the concentrations of both u and v at each grid cell
    over time.
    """

    time_steps = np.linspace(0, t, t)

    f1, k1 = 0.02,0.05
    concentration_u1 = np.load("data/concentration_u_gs_0.02_0.05_periodic_5000.npy")
    concentration_v1 = np.load("data/concentration_v_gs_0.02_0.05_periodic_5000.npy")

    f2, k2 = 0.022, 0.051
    concentration_u2 = np.load("data/concentration_u_gs_0.022_0.051_periodic_5000.npy")
    concentration_v2 = np.load("data/concentration_v_gs_0.022_0.051_periodic_5000.npy")

    f3, k3 = 0.035, 0.060
    concentration_u3 = np.load("data/concentration_u_gs_0.035_0.060_periodic_5000.npy")
    concentration_v3 = np.load("data/concentration_v_gs_0.035_0.060_periodic_5000.npy")


    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5), constrained_layout=True)

    axes[0].plot(time_steps, concentration_u1, label='$u$')
    axes[0].plot(time_steps, concentration_v1, label='$v$')
    axes[1].plot(time_steps, concentration_u2, label='$u$')
    axes[1].plot(time_steps, concentration_v2, label='$v$')
    axes[2].plot(time_steps, concentration_u3, label='$u$')
    axes[2].plot(time_steps, concentration_v3, label='$v$')

    axes[0].set_ylabel("Concentration", fontsize=18)
    axes[1].set_xlabel("$t$", fontsize=18)
    axes[0].set_title("$f$=%1.3f," %f1 + "$k$=%1.3f" %k1, fontsize=18)
    axes[1].set_title("$f$=%1.3f," %f2 + "$k$=%1.3f" %k2, fontsize=18)
    axes[2].set_title("$f$=%1.3f," %f3 + "$k$=%1.3f" %k3, fontsize=18)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=16)    

    return fig

def visualise_concentrations_noise(t):
    """
    Visualises the mean and 95% confidence interval of the concentrations 
    of both u and v at each grid cell over time when noise is added to 
    the initial state. 
    """

    time_steps = np.linspace(0, t, t)

    # Load data
    f1, k1 = 0.02,0.05
    all_concentrations_u1 = np.load("data/all_u_noise_gs_0.02_0.05_periodic_5000.npy")
    all_concentrations_v1 = np.load("data/all_v_noise_gs_0.02_0.05_periodic_5000.npy")

    f2, k2 = 0.022, 0.051
    all_concentrations_u2 = np.load("data/all_u_noise_gs_0.022_0.051_periodic_5000.npy")
    all_concentrations_v2 = np.load("data/all_v_noise_gs_0.022_0.051_periodic_5000.npy")

    f3, k3 = 0.035, 0.060
    all_concentrations_u3 = np.load("data/all_u_noise_gs_0.035_0.060_periodic_5000.npy")
    all_concentrations_v3 = np.load("data/all_v_noise_gs_0.035_0.060_periodic_5000.npy")
    
    # Calculate mean
    mean_u1 = np.mean(all_concentrations_u1, axis=0)
    mean_v1 = np.mean(all_concentrations_v1, axis=0)
    mean_u2 = np.mean(all_concentrations_u2, axis=0)
    mean_v2 = np.mean(all_concentrations_v2, axis=0)
    mean_u3 = np.mean(all_concentrations_u3, axis=0)
    mean_v3 = np.mean(all_concentrations_v3, axis=0)

    # Calculate standard deviation
    stdev_u1 = np.std(all_concentrations_u1, axis=0)
    stdev_v1 = np.std(all_concentrations_v1, axis=0)
    stdev_u2 = np.std(all_concentrations_u2, axis=0)
    stdev_v2 = np.std(all_concentrations_v2, axis=0)
    stdev_u3 = np.std(all_concentrations_u3, axis=0)
    stdev_v3 = np.std(all_concentrations_v3, axis=0)

    # Calculate 95% confidence interval
    conf_inv_u1 = 1.96 * stdev_u1 / np.sqrt(10)
    conf_inv_v1 = 1.96 * stdev_v1 / np.sqrt(10)
    conf_inv_u2 = 1.96 * stdev_u2 / np.sqrt(10)
    conf_inv_v2 = 1.96 * stdev_v2 / np.sqrt(10)
    conf_inv_u3 = 1.96 * stdev_u3 / np.sqrt(10)
    conf_inv_v3 = 1.96 * stdev_v3 / np.sqrt(10)

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5), constrained_layout=True)

    axes[0].plot(time_steps, mean_u1, label='Mean $u$')
    axes[0].fill_between(time_steps, mean_u1 - conf_inv_u1, mean_u1 + conf_inv_u1, alpha=0.4, label='95% CI')
    axes[0].plot(time_steps, mean_v1, label='Mean $v$')
    axes[0].fill_between(time_steps, mean_v1 - conf_inv_v1, mean_v1 + conf_inv_v1, alpha=0.4, label='95% CI')
    
    axes[1].plot(time_steps, mean_u2)
    axes[1].fill_between(time_steps, mean_u2 - conf_inv_u2, mean_u2 + conf_inv_u2, alpha=0.4)
    axes[1].plot(time_steps, mean_v2)
    axes[1].fill_between(time_steps, mean_v2 - conf_inv_v2, mean_v2 + conf_inv_v2, alpha=0.4)
    
    axes[2].plot(time_steps, mean_u3)
    axes[2].fill_between(time_steps, mean_u3 - conf_inv_u3, mean_u3 + conf_inv_u3, alpha=0.4)
    axes[2].plot(time_steps, mean_v3)
    axes[2].fill_between(time_steps, mean_v3 - conf_inv_v3, mean_v3 + conf_inv_v3, alpha=0.4)
    
    
    axes[0].set_ylabel("Concentration", fontsize=18)
    axes[1].set_xlabel("$t$", fontsize=18)
    axes[0].set_title("$f$=%1.3f," %f1 + "$k$=%1.3f" %k1, fontsize=18)
    axes[1].set_title("$f$=%1.3f," %f2 + "$k$=%1.3f" %k2, fontsize=18)
    axes[2].set_title("$f$=%1.3f," %f3 + "$k$=%1.3f" %k3, fontsize=18)

    fig.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=16)    

    return fig

t = 5000

fig1 = visualise_gray_scott(t)
fig2 = visualise_gray_scott_noise(t)

fig1.savefig("results/gs_5000.png", dpi=300, bbox_inches="tight")
fig2.savefig("results/gs_noise_5000.png", dpi=300, bbox_inches="tight")

fig3 = visualise_concentrations(t)
fig3.savefig("results/gs_concentrations.png", dpi=300, bbox_inches="tight")

fig4 = visualise_concentrations_noise(t)
fig4.savefig("results/gs_concentrations_noise.png", dpi=300, bbox_inches="tight")
