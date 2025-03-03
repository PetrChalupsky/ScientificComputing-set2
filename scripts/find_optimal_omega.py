"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Measures number of iterations needed for each omega and eta.
"""
from scientificcomputing_set2.optimal_omega_DLA_Laplace import search_omega_run_DLA
from scientificcomputing_set2.DLA_Laplace import create_objects
import numpy as np
import multiprocessing as mp

def run_single_simulation(args):
    """Run a single simulation with the given parameters"""
    omega, eta, width, list_objects, seed = args
    
    # Set seed
    np.random.seed(seed)
    
    # Run simulation
    cluster = create_objects(list_objects, int(width))
    k = search_omega_run_DLA(eta, omega, cluster)
    return k

def process_eta(eta, omega_list, width, list_objects, repeat_index=0):
    """Process all omega values for a single eta value"""
    
    # Prepare consistent seed
    base_seed = 42 + repeat_index * 100
    seeds = [base_seed + i for i in range(len(omega_list))]
    
    params = [(omega, eta, width, list_objects, seed) 
              for omega, seed in zip(omega_list, seeds)]
    
    # Parallelize
    with mp.Pool(processes=mp.cpu_count()) as pool:
        num_iterations = pool.map(run_single_simulation, params)
    
    return num_iterations

if __name__ == "__main__":
    width = 100
    list_objects = np.array([[2, 4, 49, 51]], dtype=np.int64)
    omega_list = np.arange(1.5, 1.92, 0.01)
    eta_list = [0.2, 1, 1.5]
    
    np.save('data/omega_list.npy', omega_list)
    num_of_repeats = 30
     
    for eta in eta_list:
        num_iterations_repeat = [None] * num_of_repeats
        for i in range(num_of_repeats):
            num_iterations_repeat[i] = process_eta(eta, omega_list, width, list_objects, repeat_index=i)
            print(f"Repeat {i+1} for eta={eta}: {num_iterations_repeat[i]}")
        
        print(f"All repeats for eta={eta}:")
        print(num_iterations_repeat)
        
        num_iterations_repeat = np.array(num_iterations_repeat)
        print(num_iterations_repeat)
        
        average = np.mean(num_iterations_repeat, axis=0)
        std = np.std(num_iterations_repeat, axis=0, ddof=1)
        print(f"Average for eta={eta}: {average}; std={std}")
        np.save(f'data/average_num_iterations_eta{eta}', average)
        np.save(f'data/std_num_iterations_eta{eta}', std)
        
