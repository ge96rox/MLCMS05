import sys
import numpy as np
from scipy.spatial.distance import cdist


def approx_nonlinear_func(x,fx,L,e):
    """
    Approximates a function using nonlinear radial functions as a basis.

    Parameters
    ----------
   
    Returns
    -------
   
    """
   
    # choose L random elements of the data
    rng = np.random.default_rng()
    x_l = rng.choice(x,L)
      
    # choose epsilon similar to diffusion map
    dist = cdist(x,x_l)
    epsilon = e * np.max(dist)
    phi_l = np.exp(-dist**2/epsilon**2)
    
    C,residues,_,_ = np.linalg.lstsq(phi_l, fx, rcond=50000)
    
    
    return C,residues,epsilon,x_l
