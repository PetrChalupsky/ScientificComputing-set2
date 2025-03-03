"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Visualize optimal omega for DLA with SOR.
"""
import matplotlib.pyplot as plt
import numpy as np

eta_list = [0.2,1,1.5]
omega_list = np.load('data/omega_list.npy')

for eta in eta_list:
    num_iterations = np.load(f'data/average_num_iterations_eta{eta}.npy')
    std = np.load(f'data/std_num_iterations_eta{eta}.npy')
    plt.plot(omega_list, num_iterations, label=f'$\eta$ = {eta}')
    plt.fill_between(omega_list, (num_iterations - 2*std), (num_iterations + 2*std), alpha=0.3)

plt.legend()
plt.xlabel('$\omega$')
plt.ylabel('Total number of iterations')
plt.savefig('results/DLA_Laplace_optimal_omega', dpi=300)
