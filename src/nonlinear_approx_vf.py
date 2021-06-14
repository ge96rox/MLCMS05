import numpy as np
from scipy.spatial.distance import cdist
from src.nonlinear_approx import approx_nonlinear_func


# rbf function
def rbf(x, x_l, eps):
    #return np.exp(-np.sum((x_l - x) ** 2, axis=-1) / (eps ** 2))
    return np.exp(-cdist(x,x_l) ** 2 / eps ** 2)


# return random index
def rand_idx(x, nr_randpts):
    return np.random.permutation(x.shape[0])[0:nr_randpts]


# return phi
def get_phi(x0_data, nr_xl, id_xl, current_x_data, eps):
    # nr = x0_data.shape[0]
    nr = current_x_data.shape[0]
    phi = np.empty((nr, nr_xl))
    '''
    for i in range(nr_xl):
        l = id_xl[i]
        phi[:, i] = rbf(current_x_data, x0_data[l], eps)
    '''
    phi = rbf(current_x_data, x0_data[id_xl], eps)
    return phi


def find_best_eps(x0_data, nr_xl, delta_t, x1_data, v_data):
    min_loss = np.inf
    min_esp = 0
    eps = np.linspace(0.1, 20, 20)
    id_xl = rand_idx(x0_data, nr_xl)
    for e in eps:
        phi = get_phi(x0_data, nr_xl, id_xl, x0_data, e)
        C = np.linalg.lstsq(phi, v_data, rcond=None)[0]
        x1_sol = np.zeros((x0_data.shape[0], 2))
        v_sol = phi @ C
        x1_sol = v_sol * delta_t + x0_data
        loss = np.mean(np.sum((x1_sol - x1_data) ** 2, axis=-1))
        if loss < min_loss:
            min_loss = loss
            min_esp = e
    return min_esp
