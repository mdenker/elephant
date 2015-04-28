import scipy.special
import scipy as sp
import scipy.sparse
import numpy as np
import time
import itertools
import quantities as pq
import neo
import warnings


def hash(m, N, base=2):
    """
    Calculate for a spike pattern or a matrix of spike patterns (provide each pattern as a column)
    composed of N neurons a unique number.
    
    
    Parameters:
    -----------
    m [int. | iterable]:
           matrix of 0-1 patterns as columns, shape: (number of neurons, number of patterns)
    N [int. ]:
           number of neurons is required to be equal to the number of rows
    base [int. default to 2]: 
    ### TODO:  write the technical number
           base for calculation of the exponentialsequen
    
    Returns:
    --------
    list of integers:
           An array containing the hash values of each pattern,
           shape: (number of patterns)
    
    Raises:
    -------
       ValueError: if matrix m has wrong orientation
    
    Examples:
    ---------
    descriptive example:
    m = [0
         1
         1]
    N = 3
    base = 2
    hash = 0*2^2 + 1*2^1 + 1*2^0 = 3

    second example:
    >>> import numpy as np
    >>> m = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
                         [0, 0, 1, 0, 1, 0, 1, 1],
                         [0, 0, 0, 1, 0, 1, 1, 1]])
    
    >>> hash(m,N=3)
    Out[1]: array([0, 4, 2, 1, 6, 5, 3, 7])
    """
    # check the consistency between shape of m and number neurons N
    if N != np.shape(m)[0]:
        raise ValueError('patterns in the matrix should be column entries')


    # generate the representation for binary system
    #XXX what happens if the pattern is longer than 63?
    v = np.array([base**x for x in range(N)])
    # reverse the order
    v = v[np.argsort(-v)]
    # calculate the binary number by use of scalar product
    return np.dot(v,m)

def inv_hash(h,N,base=2):
    """
    Calculate the 0-1 spike patterns (matrix) from hash values
    
    Parameters:
    -----------
    h [int. | iterable]:
           list or array of hash values, length: number of patterns
    N [int.]:
           number of neurons
    base [int. default to 2]: 
           ### TODO: write the proper doc
           base for calculation of the exponential
           
    Raises:
    -------
       ValueError: if the hash is not compatible with the number of neurons
       hash value should not be larger than the biggest possible hash number with 
       given number of neurons
       (e.g. for N = 2, max(hash) = 2^1 + 2^0 = 3
            , or for N = 4, max(hash) = 2^3 + 2^2 + 2^1 + 2^0 = 15)
    
    Returns:
    --------
       numpy.array:
           A matrix of shape: (N, number of patterns)

    Examples
    ---------
    >>> import numpy as np
    >>> h = np.array([3,7])
    >>> N = 4
    >>> inv_hash(h,N)
    Out[1]: 
    array([[1, 1],
           [1, 1],
           [0, 1],
           [0, 0]])
    """
    
    # check if the hash values are not bigger than possible hash value
    # for N neuron with basis = base
    if np.any(h > np.sum([base**x for x in range(N)])):
        raise ValueError("hash value is not compatible with the number of neurons N")
    # check if the hash values are integer
    if np.all(np.int64(h) == h) == False:
        raise ValueError("hash values are not integers")

    m = np.zeros((N,len(h)), dtype=int) 
    for  j, hh in enumerate(h):
        i = N-1
        while i>=0 and hh != 0:
            m[i,j] = hh % base
            hh = hh / base
            i-=1
    return m        

def N_emp_mat(mat,N, pattern_hash,base=2):
    """
    Calculates empirical number of observed patterns expressed by their hash values
    
    Parameters:
    -----------
    m [int. | iterable]:
           matrix of 0-1 patterns as columns, shape: (number of neurons N, number of patterns)
    N [int.]:
           number of neurons
    pattern_hash [int. | iterable ]:
            array of hash values. Length defines number of patterns
    
    base 
    Returns:
    --------
    N_emp [int. | iterable]:
           empirical number of each observed pattern. Same length as pattern_hash 
    indices [list of list | iterable]:
           list of indices of mat per entry of pattern_hash. indices[i] = N_emp[i] = pattern_hash[i] 

    Raises:
    -------
       ValueError: if mat is not zero-one matrix
       
    Examples:
    ---------
    >>> mat = np.array([[1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 0]])
    >>> pattern_hash = np.array([1,3])
    >>> n_emp, n_emp_indices = N_emp_mat(mat, N,pattern_hash)
    >>> print n_emp
    [ 1.  2.]
    >>> print n_emp_indices
    [array([4]), array([0, 3])]
    """
    # check if the mat is zero-one matrix
    if np.any(mat>1) or np.any(mat<0):
        raise "ValueError: entries of mat should be either one or zero"
    h = hash(mat,N,base = base)
    N_emp = np.zeros(len(pattern_hash))
    indices = []
    for p_h_idx, p_h in enumerate(pattern_hash):
        indices_tmp = np.nonzero(h == p_h)[0]
        indices.append(indices_tmp)
        N_emp_tmp = len(indices_tmp)
        N_emp[p_h_idx] = N_emp_tmp
    return N_emp, indices

