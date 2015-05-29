import scipy.special
import scipy as sp
import scipy.sparse
import numpy as np
import time
import itertools
import quantities as pq
import neo
import warnings
import elephant.conversion as conv


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
           base for calculation of the number from binary sequences (= pattern)
    
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
           base for calculation of the number from binary sequences (= pattern)
           
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

def n_emp_mat(mat,N, pattern_hash,base=2):
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
    [ 0.  2.]
    >>> print n_emp_indices
    [array([]), array([0, 3])]
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


def n_emp_mat_sum_trial(mat, N,pattern_hash):
    """
    Calculates empirical number of observed patterns expressed summed across trials
    
    Parameters:
    -----------
    mat [zero or one matrix]:
            0-axis --> trials
            1-axis --> neurons
            2-axis --> time bins
    N [int.]:
           number of neurons
    pattern_hash [int. | iterable ]:
            array of hash values, length: number of patterns
    

    Returns:
    --------
    N_emp [int. | iterable]:
           empirical number of observed pattern summed across trials, length: number of patterns (i.e. len(patter_hash))
    idx_trials [list of list | iterable]:
           list of indices of mat for each trial in which the specific pattern has been observed.
           0-axis --> trial
           1-axis --> list of indices for the chosen trial per entry of pattern_hash
    
    Raises:
    -------
       ValueError: if matrix mat has wrong orientation
       ValueError: if mat is not zero-one matrix
       
    Examples:
    ---------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([4,6])
    >>> N = 3
    >>> n_emp_sum_trial, n_emp_sum_trial_idx = n_emp_mat_sum_trial(mat, N,pattern_hash)
    >>> n_emp_sum_trial
    Out[0]: array([ 1.,  3.])
    >>> n_emp_sum_trial_idx
    Out[1]: [[array([0]), array([3])], [array([], dtype=int64), array([2, 4])]]
    """
    # check the consistency between shape of m and number neurons N
    if N != np.shape(mat)[1]:
        raise ValueError('the entries of mat should be a list of a list where 0-axis is trials and 1-axis is neurons')


    num_patt = len(pattern_hash)
    N_emp = np.zeros(num_patt)

    idx_trials = []
    for mat_tr in mat:
        # check if the mat is zero-one matrix
        if np.any(np.array(mat_tr)>1):
            raise "ValueError: entries of mat should be either one or zero"
        N_emp_tmp,indices_tmp = n_emp_mat(mat_tr,N, pattern_hash,base=2)
        idx_trials.append(indices_tmp)
        N_emp += N_emp_tmp
    return N_emp, idx_trials


def _sts_overlap(sts, t_start = None, t_stop = None):
    """
    Find the internal range t_start, t_stop where all spike trains are
    defined; cut all spike trains taking that time range only
    """
    max_tstart = max([t.t_start for t in sts])
    min_tstop = min([t.t_stop for t in sts])

    if t_start is None:
        t_start = max_tstart
        if not all([max_tstart == t.t_start for t in sts]):
            warnings.warn(
                "Spiketrains have different t_start values -- "
                "using maximum t_start as t_start.")

    if t_stop is None:
        t_stop = min_tstop
        if not all([min_tstop == t.t_stop for t in sts]):
            warnings.warn(
                "Spiketrains have different t_stop values -- "
                "using minimum t_stop as t_stop.")

    sts_cut = [st.time_slice(t_start=t_start, t_stop=t_stop) for st in sts]
    return sts_cut


def n_exp_mat(mat, N,pattern_hash, method = 'anal', **kwargs):
    """
    Calculates the expected joint probability for each spike pattern
    
    Parameters:
    -----------
    mat [zero or one matrix]:
            0-axis --> neurons
            1-axis --> time bins
    pattern_hash [int. | iterable ]:
            array of hash values, length: number of patterns
    method [string | default 'anal']:
            method with which the expectency should be caculated
            'anal' -- > analytically
            'surr' -- > with surrogates
    kwargs:
    -------
    n_surr [int. default to 3000]:
            number of surrogate to be used

    
    Raises:
    -------
       ValueError: if matrix m has wrong orientation

    Returns:
    --------
    if method is anal:
        numpy.array:
           An array containing the expected joint probability of each pattern,
           shape: (number of patterns,)
    if method is surr:
        numpy.ndarray, 0-axis --> different realizations, length = number of surrogates
                       1-axis --> patterns

    Examples:
    ---------
    >>> mat = array([[1, 1, 1, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0]])
    >>> pattern_hash = np.array([5,6])
    >>> N = 3
    >>> n_exp_anal = n_exp_mat(mat,N, pattern_hash, method = 'anal')
    >>> n_exp_anal
    Out[0]: [ 0.5 1.5 ]
    
    >>> 
    >>>
    >>> n_exp_surr = n_exp_mat(mat, N,pattern_hash, method = 'surr', n_surr = 5000)
    >>> print n_exp_surr
    [[ 1.  1.]
     [ 2.  0.]
     [ 2.  0.]
     ..., 
     [ 2.  0.]
     [ 2.  0.]
     [ 1.  1.]]
     
    """
    # check if the mat is zero-one matrix
    if np.any(mat>1) or np.any(mat<0):
        raise "ValueError: entries of mat should be either one or zero"
    
    if method == 'anal':
        marg_prob = np.mean(mat,1,dtype=float)
        #marg_prob needs to be a column vector, so we 
        #build a two dimensional array with 1 column 
        #and len(marg_prob) rows
        marg_prob = np.reshape(marg_prob,(len(marg_prob),1))
        m = inv_hash(pattern_hash, N)
        nrep = np.shape(m)[1] 
        # multipyling the marginal probability of neurons with regard to the pattern 
        pmat = np.multiply(m,np.tile(marg_prob,(1,nrep)))+ np.multiply(1-m,np.tile(1-marg_prob,(1,nrep)))
        return np.prod(pmat,axis=0)*float(np.shape(mat)[1])
    if method == 'surr':
        if 'n_surr' in kwargs: 
            n_surr = kwargs['n_surr'] 
        else: 
            n_surr = 3000.
        N_exp_array = np.zeros((n_surr,len(pattern_hash)))
        for rz_idx, rz in enumerate(np.arange(n_surr)):
            # shuffling all elements of zero-one matrix
            [np.random.shuffle(i) for i in mat]
            N_exp_array[rz_idx] = N_emp(mat, pattern_hash)[0]
        return N_exp_array




def n_exp_mat_sum_trial(mat,N, pattern_hash, method = 'anal'):
    """
    Calculates the expected joint probability for each spike pattern sum over trials
    
    Parameters:
    -----------
    mat [zero or one matrix]:
            0-axis --> trials
            1-axis --> neurons
            2-axis --> time bins
    N [int.]:
           number of neurons
    pattern_hash [int. | iterable ]:
            array of hash values, length: number of patterns
    

    Returns:
    --------
    if method is anal:
        numpy.array:
           An array containing the expected joint probability of each pattern summed over trials,
           shape: (number of patterns,)

    Examples:
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([5,6])
    >>> N = 3
    >>> n_exp_anal = n_exp_mat_sum_trial(mat, N, pattern_hash, method = 'anal')
    >>> print n_exp_anal
    Out[0]: array([ 1.56,  2.56])
    """
    # check the consistency between shape of m and number neurons N
    if N != np.shape(mat)[1]:
        raise ValueError('the entries of mat should be a list of a list where 0-axis is trials and 1-axis is neurons')

    if method == 'anal':
        n_exp = np.zeros(len(pattern_hash))
        for mat_tr in mat:
            n_exp += n_exp_mat(mat_tr, N,pattern_hash, method = 'anal')
    else:
        raise ValueError(
            "The method only works on the zero_one matrix at the moment")

    return n_exp        
