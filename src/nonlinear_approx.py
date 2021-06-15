import sys
import numpy as np
from scipy.spatial.distance import cdist


# rbf function
def rbf(x, x_l, eps):
    return np.exp(-cdist(x, x_l) ** 2 / eps ** 2)


# return random index
def rand_idx(x, nr_randpts):
    return np.random.permutation(x.shape[0])[0:nr_randpts]


# return phi
def get_phi(x0_data, id_xl, current_x_data, eps):
    phi = rbf(current_x_data, x0_data[id_xl], eps)
    return phi


def approx_nonlinear_func(x, fx, L, e):
    """
    Approximates a function using nonlinear radial functions as a basis.

    Parameters
    ----------
   
    Returns
    -------
   
    """

    # choose L random elements of the data
    id_xl = rand_idx(x, L)

    # choose epsilon similar to diffusion map
    dist = cdist(x, x[id_xl])
    epsilon = e * np.max(dist)
    # phi_l = np.exp(-dist**2/epsilon**2)
    phi = get_phi(x, id_xl, x, epsilon)

    C, res, _, _ = np.linalg.lstsq(phi, fx, rcond=50000)

    def approximated_func(x_new):
        dist_new = cdist(x_new, x[id_xl])
        phi_new = np.exp(-dist_new ** 2 / epsilon ** 2)

        return phi_new @ C

    return approximated_func, C, res, epsilon

