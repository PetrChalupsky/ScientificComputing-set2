import matplotlib.pyplot as plt
import numpy as np




eta_list = [0.2,1,1.5]
omega_list = np.load('data/omega_list.npy')
for eta in eta_list:
    num_iterations = np.load(f'data/num_iterations_eta{eta}.npy')
    plt.plot(omega_list, num_iterations, label=f'$\eta$ = {eta}')

plt.legend()
plt.xlabel('$\omega$')
plt.ylabel('Total number of iterations')
plt.savefig('results/DLA_Laplace_optimal_omega', dpi=300)
