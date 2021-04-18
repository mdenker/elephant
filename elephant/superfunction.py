import numpy as np

def Gfunction(ti):
    r"""
    Calculate G for a sequence of intervals. G is an useful measure to identify the statistics of the spike train,

    Given a vector :math:`I` containing a sequence of intervals:
    .. math::
        G := \frac{1}{N} \sum_{i=1}^{N-1}
                          \frac{3(I_i-I_{i+1})^2}
                          {(I_i+I_{i+1})^2}



    Parameters
    ----------
    ti - consecutive time interval


    Returns
    -------
    float


    Examples
    --------
    >>> from elephant import superfunction
    >>> superfunction.Gfunction([0.3, 4.5, 6.7, 9.3])
    0.6383154236534495

    """
    # convert to array
    ti = np.asarray(ti)

    G = []
    for i in range(len(ti) - 1):
        x= ti[i + 1] - ti[i] # upper
        y= ti[i] + ti[i+1] # lower
        G.append(x/y)

    G_2 = []
    for i in range(len(G)):
        G_2.append((G[i] * G[i]) * 3)

    return np.mean(G_2)

