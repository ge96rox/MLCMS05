import sys
import numpy as np
from scipy.spatial.distance import cdist


def approx_linear_func(x,fx):
    """
    Approximates a function using np.linalg.lstsq

    Parameters
    ----------
    x: np.ndarray
        source data
    fx: np.ndarray
        target data
    Returns
    -------
    approximated_func:
        function closure that can be used to approximate
    A:
        matrix C
    res:
       residual from lstsq method
    """
   
       
    A,res,_,_ = np.linalg.lstsq(x, fx, rcond=1e-16)

    def approximated_func(x_new):
        return x_new@A

    return approximated_func,A,res
