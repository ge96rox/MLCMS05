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
    
    C,res,_,_ = np.linalg.lstsq(phi_l, fx, rcond=50000)
    
    def approximated_func(x_new):
        dist_new = cdist(x_new,x_l)
        phi_new = np.exp(-dist_new**2/epsilon**2)
    
        return phi_new@C
    

    
    return approximated_func,C,res,epsilon

def find_best_eps(x,fx,L,e_list):
    
    r_list = np.zeros(len(e_list))
    eps_list =[]
    c_list = []
    func_list = []
    for i,e in enumerate(e_list):

        approximated_func,C,res,epsilon = approx_nonlinear_func(x,fx,L,e)
        r_list[i] = res[0] if res.size!= 0 else float("inf")
        eps_list.append(epsilon)
        c_list.append(C)
        func_list.append(approximated_func)

    print("Minimum Residual is :", min(r_list))
    print("At e = " ,e_list[np.argmin(r_list)])

    epsilon = eps_list[np.argmin(r_list)]
    nonlinear_func = func_list[np.argmin(r_list)]
            
    return nonlinear_func, epsilon