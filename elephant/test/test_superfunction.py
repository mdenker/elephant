from __future__ import division

import itertools
import math
import unittest
from elephant.superfunction import Gfunction as Gf

import neo
import numpy as np
import quantities as pq
import scipy.integrate as spint
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_array_less


class GfTestCase(unittest.TestCase):
    def test_G(self):
        test_seq = [1, 28, 4, 47, 5, 16, 2, 5, 21, 12,
                         4, 12, 59, 2, 4, 18, 33, 25, 2, 34,
                         4, 1, 1, 14, 8, 1, 10, 1, 8, 20,
                         5, 1, 6, 5, 12, 2, 8, 8, 2, 8,
                         2, 10, 2, 1, 1, 2, 15, 3, 20, 6,
                         11, 6, 18, 2, 5, 17, 4, 3, 13, 6,
                         1, 18, 1, 16, 12, 2, 52, 2, 5, 7,
                         6, 25, 6, 5, 3, 15, 4, 3, 16, 3,
                         6, 5, 24, 21, 3, 3, 4, 8, 4, 11,
                         5, 7, 5, 6, 8, 11, 33, 10, 7, 4]

        target = 0.971826029994

        assert_array_almost_equal(Gf(test_seq), target, decimal=9)
