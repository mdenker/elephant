"""
Various functions to test the ue_utils package

@author: Rostami
"""

import unittest
import numpy as np
import quantities as pq
import elephant.ue_utils as ue


class UETestCase(unittest.TestCase):
    
    def test_hash_default_orientation_col(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        expected = np.array([77,43,23])
        h = ue.hash(m,orientation = "col")
        self.assertTrue(np.all(expected == h))

    def test_hash_default_orientation_row(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        expected = np.array([0,4,2,1,6,5,3,7])
        h = ue.hash(m,orientation = "row")
        self.assertTrue(np.all(expected == h))

    def test_hash_default_longpattern(self):
        m = np.zeros((100,2))
        m[0,0] = 1
        expected = np.array([2**99,0])
        h = ue.hash(m,orientation = "col")
        self.assertTrue(np.all(expected == h))

    def test_hash_ValueError(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.hash,m,'cols')

    def test_hash_base_not_two(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        base = 3
        expected = np.array([0,9,3,1,12,10,4,13])
        h = ue.hash(m,orientation = 'row',base=base)
        self.assertTrue(np.all(expected == h))

    ## TODO: write a test for ValueError in inv_hash
    def test_invhash_ValueError(self):
        self.assertRaises(ValueError,ue.inv_hash,[128,8],4)

    def test_invhash_default_base(self):
        N = 3
        h = np.array([0, 4, 2, 1, 6, 5, 3, 7])
        expected = np.array([[0, 1, 0, 0, 1, 1, 0, 1],[0, 0, 1, 0, 1, 0, 1, 1],[0, 0, 0, 1, 0, 1, 1, 1]])
        m = ue.inv_hash(h,N)
        self.assertTrue(np.all(expected == m))

    def test_invhash_base_not_two(self):
        N = 3
        h = np.array([1,4,13])
        base = 3
        expected = np.array([[0,0,1],[0,1,1],[1,1,1]])
        m = ue.inv_hash(h,N,base)
        self.assertTrue(np.all(expected == m))

    def test_invhash_shape_mat(self):
        N = 8
        h = np.array([178, 212, 232])
        expected = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        m = ue.inv_hash(h,N)
        self.assertTrue(np.shape(m)[0] == N)

    def test_hash_invhash_consistency(self):
        m = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1]])
        inv_h = ue.hash(m,orientation = 'col')
        m1 = ue.inv_hash(inv_h, N = 8)
        self.assertTrue(np.all(m == m1))

    def test_hash_invhash_consistency_orientation_rows(self):
        m = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1]])
        inv_h = ue.hash(m,orientation = 'row')
        m1 = ue.inv_hash(inv_h, N = 3)
        self.assertTrue(np.all(m.T == m1))

def suite():
    suite = unittest.makeSuite(UETestCase, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

