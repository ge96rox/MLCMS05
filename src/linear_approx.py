import sys
import numpy as np
from scipy.spatial.distance import cdist


def approx_linear_func(x,fx):
    """
    Approximates a function using using np.linalg.lstsq 

    Parameters
    ----------
   
    Returns
    -------
   
    """
   
       
    A,res,_,_ = np.linalg.lstsq(x, fx, rcond=500000)

    def approximated_func(x_new):
        return x_new@A

    return approximated_func,A,res
