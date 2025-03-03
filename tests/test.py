import unittest
import numpy as np
from scientificcomputing_set2.gray_scott import update_grids_dirichlet
from scientificcomputing_set2.DLA_Laplace import initialize_grid, create_objects

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


    def test_initialize_grid(self):
        expected_grid = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
        actual_grid = initialize_grid(3, False)
        self.assertTrue(np.array_equal(expected_grid, actual_grid))

    def test_create_object(self):
        list_objects = np.array([[1,3,1,3]])
        width = 3
        expected_grid = [[0,0,0],[0,1,1],[0,1,1]]
        actual_grid = create_objects(list_objects, width)
        self.assertTrue(np.array_equal(expected_grid, actual_grid))


if __name__ == "__main__":
    unittest.main()
