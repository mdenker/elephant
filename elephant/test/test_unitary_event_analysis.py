"""
Various functions to test the ue_utils package

@author: Rostami
"""

import unittest
import numpy as np
import quantities as pq
import types
import elephant.unitary_event_analysis as ue


class UETestCase(unittest.TestCase):

    def test_hash_default(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        expected = np.array([77,43,23])
        h = ue.hash_from_pattern(m, N=8)
        self.assertTrue(np.all(expected == h))

    def test_hash_default_longpattern(self):
        m = np.zeros((100,2))
        m[0,0] = 1
        expected = np.array([2**99,0])
        h = ue.hash_from_pattern(m, N=100)
        self.assertTrue(np.all(expected == h))

    def test_hash_ValueError(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError, ue.hash_from_pattern, m, N=3)

    def test_hash_base_not_two(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        m = m.T
        base = 3
        expected = np.array([0,9,3,1,12,10,4,13])
        h = ue.hash_from_pattern(m, N=3, base=base)
        self.assertTrue(np.all(expected == h))

    ## TODO: write a test for ValueError in inverse_hash_from_pattern
    def test_invhash_ValueError(self):
        self.assertRaises(ValueError, ue.inverse_hash_from_pattern, [128, 8], 4)

    def test_invhash_default_base(self):
        N = 3
        h = np.array([0, 4, 2, 1, 6, 5, 3, 7])
        expected = np.array([[0, 1, 0, 0, 1, 1, 0, 1],[0, 0, 1, 0, 1, 0, 1, 1],[0, 0, 0, 1, 0, 1, 1, 1]])
        m = ue.inverse_hash_from_pattern(h, N)
        self.assertTrue(np.all(expected == m))

    def test_invhash_base_not_two(self):
        N = 3
        h = np.array([1,4,13])
        base = 3
        expected = np.array([[0,0,1],[0,1,1],[1,1,1]])
        m = ue.inverse_hash_from_pattern(h, N, base)
        self.assertTrue(np.all(expected == m))

    def test_invhash_shape_mat(self):
        N = 8
        h = np.array([178, 212, 232])
        expected = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        m = ue.inverse_hash_from_pattern(h, N)
        self.assertTrue(np.shape(m)[0] == N)

    def test_hash_invhash_consistency(self):
        m = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1]])
        inv_h = ue.hash_from_pattern(m, N=8)
        m1 = ue.inverse_hash_from_pattern(inv_h, N = 8)
        self.assertTrue(np.all(m == m1))

    def test_n_emp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 1, 1, 1],[1, 0, 1, 1, 1]])
        N = 4
        pattern_hash = [3, 15]
        expected1 = np.array([ 2.,  1.])
        expected2 = [[0, 2], [4]]
        nemp,nemp_indices = ue.n_emp_mat(mat,N,pattern_hash)
        self.assertTrue(np.all(nemp == expected1))
        for item_cnt,item in enumerate(nemp_indices):
            self.assertTrue(np.allclose(expected2[item_cnt],item))

    def test_n_emp_mat_sum_trial_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([4,6])
        N = 3
        expected1 = np.array([ 1.,  3.])
        expected2 = [[[0], [3]],[[],[2,4]]]
        n_emp, n_emp_idx = ue.n_emp_mat_sum_trial(mat, N,pattern_hash)
        self.assertTrue(np.all(n_emp == expected1))
        for item0_cnt,item0 in enumerate(n_emp_idx):
            for item1_cnt,item1 in enumerate(item0):
                self.assertTrue(np.allclose(expected2[item0_cnt][item1_cnt],item1))

    def test_n_emp_mat_sum_trial_ValueError(self):
        mat = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.n_emp_mat_sum_trial,mat,N=2,pattern_hash = [3,6])

    def test_n_exp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 1, 1, 1],[1, 0, 1, 1, 1]])
        N = 4
        pattern_hash = [3, 11]
        expected = np.array([ 1.536,  1.024])
        nexp = ue.n_exp_mat(mat,N,pattern_hash)
        self.assertTrue(np.allclose(expected,nexp))

    def test_n_exp_mat_sum_trial_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.56,  2.56])
        n_exp = ue.n_exp_mat_sum_trial(mat, N,pattern_hash)
        self.assertTrue(np.allclose(n_exp,expected))

    def test_n_exp_mat_sum_trial_TrialAverage(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.62,  2.52])
        n_exp = ue.n_exp_mat_sum_trial(mat, N,pattern_hash,method = 'analytic_TrialAverage')
        self.assertTrue(np.allclose(n_exp,expected))

    def test_n_exp_mat_sum_trial_ValueError(self):
        mat = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.n_exp_mat_sum_trial,mat,N=2,pattern_hash = [3,6])

    def test_gen_pval_anal_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.56,  2.56])
        pval_func,n_exp = ue.gen_pval_anal(mat, N,pattern_hash)
        self.assertTrue(np.allclose(n_exp,expected))
        self.assertTrue(isinstance(pval_func, types.FunctionType))

    def test_jointJ_default(self):
        p_val = np.array([0.31271072,  0.01175031])
        expected = np.array([0.3419968 ,  1.92481736])
        self.assertTrue(np.allclose(ue.jointJ(p_val),expected))

    def test__rate_mat_avg_trial_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        expected = [0.9, 0.7,0.6]
        self.assertTrue(np.allclose(expected,ue._rate_mat_avg_trial(mat)))

    def test__bintime(self):
        t = 13*pq.ms
        binsize = 3*pq.ms
        expected = 4
        self.assertTrue(np.allclose(expected,ue._bintime(t,binsize)))
    def test__winpos(self):
        t_start = 10*pq.ms
        t_stop = 46*pq.ms
        winsize = 15*pq.ms
        winstep = 3*pq.ms
        expected = [ 10., 13., 16., 19., 22., 25., 28., 31.]*pq.ms
        self.assertTrue(
            np.allclose(
                ue._winpos(
                    t_start, t_stop, winsize,
                    winstep).rescale('ms').magnitude,
                expected.rescale('ms').magnitude))

    def test__UE(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([4,6])
        N = 3
        expected_S = np.array([-0.26226523,  0.04959301])
        expected_idx = [[[0], [3]], [[], [2, 4]]]
        expected_nemp = np.array([ 1.,  3.])
        expected_nexp = np.array([ 1.04,  2.56])
        expected_rate = np.array([ 0.9,  0.7,  0.6])
        S, rate_avg, n_exp, n_emp,indices = ue._UE(mat,N,pattern_hash)
        self.assertTrue(np.allclose(S ,expected_S))
        self.assertTrue(np.allclose(n_exp ,expected_nexp))
        self.assertTrue(np.allclose(n_emp ,expected_nemp))
        self.assertTrue(np.allclose(expected_rate ,rate_avg))
        for item0_cnt,item0 in enumerate(indices):
            for item1_cnt,item1 in enumerate(item0):
                self.assertTrue(np.allclose(expected_idx[item0_cnt][item1_cnt],item1))

def suite():
    suite = unittest.makeSuite(UETestCase, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

