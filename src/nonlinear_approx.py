import numpy as np
from scipy.spatial.distance import cdist


def rbf(x, x_l, eps):
    """radial basic function

    Parameters
    ----------
    x: np.ndarray
        data
    x_l: np.ndarray
        random selected data
    eps: float
        epsilon
    Returns
    -------
    matrix contains radial basic function value
    """
    return np.exp(-cdist(x, x_l) ** 2 / eps ** 2)


def rand_idx(x, nr_randpts):
    """function returns random selected index

    Parameters
    ----------
    x: np.ndarray
        data
    nr_randpts: int
        number of points to be randomly selected
    eps: float
        epsilon
    Returns
    -------
    random selected index
    """
    return np.random.permutation(x.shape[0])[0:nr_randpts]


def get_phi(x0_data, id_xl, current_x_data, eps):
    """function returns phi

    Parameters
    ----------
    x0_data: np.ndarray
        original data
    id_xl: np.ndarray
        index of random selected data
    current_x_data: np.ndarray
        current data (in most case it should be the same to x0_data)
    eps: float
        epsilon
    Returns
    -------
    matrix contains Phi
    """
    phi = rbf(current_x_data, x0_data[id_xl], eps)
    return phi


def approx_nonlinear_func(x, fx, L, e):
    """
    Approximates a function using nonlinear radial functions as a basis.

    Parameters
    ----------
    x: np.ndarray
        source data
    fx: np.ndarray
        target data
    L: int
        hyperparameter L for RBF kernel
    e: float
        hyperparameter for choosing epsilon
    Returns
    -------
    approximated_func:
        function closure that can be used to approximate
    C:
        matrix C
    res:
       residual from lstsq method
    epsilon:
        epsilon
    """

    # choose L random elements of the data
    id_xl = rand_idx(x, L)

    # choose epsilon similar to diffusion map
    dist = cdist(x, x[id_xl])
    epsilon = e * np.max(dist)
    # phi_l = np.exp(-dist**2/epsilon**2)
    phi = get_phi(x, id_xl, x, epsilon)

    C, res, _, _ = np.linalg.lstsq(phi, fx, rcond=1e-16)

    def approximated_func(x_new):
        dist_new = cdist(x_new, x[id_xl])
        phi_new = np.exp(-dist_new ** 2 / epsilon ** 2)

        return phi_new @ C

    return approximated_func, C, res, epsilon

