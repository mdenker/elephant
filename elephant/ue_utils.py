import scipy.special
import scipy as sp
import scipy.sparse
import numpy as np
import time
import itertools
import quantities as pq
import neo
import warnings


def hash(m, orientation, base=2):
    """
    Calculate for a spike pattern or a matrix of spike patterns (provide each pattern as a column)
    composed of N neurons a unique number.
    
    
    Parameters:
    -----------
    m [int. | iterable]:
           matrix of 0-1 patterns as columns, shape: (number of neurons, number of patterns)
    orientation [string | "row" or "col" ]:
           "row": orientation of patterns in the matrix m along the rows
           "col": orientation of patterns in the matrix m along the columns
    
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
    orientation = 'col'
    base = 2
    hash = 0*2^2 + 1*2^1 + 1*2^0 = 3

    second example:
    >>> import numpy as np
    >>> m = array([[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [1, 1, 1]])
    >>> hash(m,orientation = 'row')
    Out[1]: array([0, 4, 2, 1, 6, 5, 3, 7])
    """
    # check the consistency between shape of m and number neurons N
    if orientation == "row":
        N = np.shape(m)[1]
        m = m.T
    elif orientation == "col":
        N = np.shape(m)[0]
    else:
        raise ValueError('provide "col" or "row" as orientation of pattern representation in m' )

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
           An matrix of shape: (N, number of patterns)

    Examples:
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


