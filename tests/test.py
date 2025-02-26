import unittest
import numpy as np
from scientificcomputing_set2.gray_scott import update_grids_dirichlet, update_grids_periodic

class Test(unittest.TestCase):
    def test_update_grids_dirichlet(self):
        initial_u = np.array([[0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5]])
        
        initial_v = np.array([[0, 0, 0],
                     [0, 0.25, 0],
                     [0, 0, 0]])
        
        expected_u = np.array([[0.5, 0.5, 0.5],
                               [0.5, 0.51875, 0.5],
                               [0.5, 0.5, 0.5]])
        actual_u = update_grids_dirichlet(3, 1, 1, initial_u, initial_v, 0.5, 0.125, 0.1, 0.5)[0]


        expected_v = np.array([[0, 0, 0],
                               [0, 0.00625, 0],
                               [0, 0, 0]])
        actual_v = update_grids_dirichlet(3, 1, 1, initial_u, initial_v, 0.5, 0.125, 0.1, 0.5)[1]

        self.assertTrue(np.allclose(expected_u, actual_u))
        self.assertTrue(np.allclose(expected_v, actual_v))

if __name__ == "__main__":
    unittest.main()