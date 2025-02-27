from scientificcomputing_set2.optimal_omega_DLA_Laplace import search_omega_run_DLA, create_objects
import numpy as np

width = 100
list_objects = np.array([[2,4,49, 51]], dtype=np.int64)
cluster = create_objects(list_objects,int(width))

omega_list = np.arange(1.5, 1.92, 0.01)
eta_list = [0.2,1,1.5]

for eta in eta_list:
    num_iterations = []
    for omega in omega_list:
        cluster = create_objects(list_objects,int(width))
        k = search_omega_run_DLA(eta, omega, cluster)
        num_iterations.append(k)
     
    np.save(f'data/num_iterations_eta{eta}.npy', num_iterations)
# Find index of minimal number of iterations

np.save('data/omega_list.npy', omega_list)

