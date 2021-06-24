import numpy as np
from src.nonlinear_approx import rand_idx, get_phi, approx_nonlinear_func
from scipy.integrate import solve_ivp


def find_best_eps(x, fx, L, e_list):
    """function returns epsilon with minimal mse loss for function approximation

    Parameters
    ----------
    x: np.ndarray
        source data
    fa: np.ndarray
        target data
    L: int
        hyperparameter L in RBF kernel
    e_list: list
        list of epsilon
    Returns
    -------
    nonlinear_func:
        best nonlinear function to approximate
    epsilon:
        best epsilon with minimal mse loss
    """
    r_list = np.zeros(len(e_list))
    eps_list = []
    c_list = []
    func_list = []
    for i, e in enumerate(e_list):
        approximated_func, C, res, epsilon = approx_nonlinear_func(x, fx, L, e)
        r_list[i] = res[0] if res.size != 0 else float("inf")
        eps_list.append(epsilon)
        c_list.append(C)
        func_list.append(approximated_func)

    print("Minimum Residual is :", min(r_list))
    print("At e = ", e_list[np.argmin(r_list)])

    epsilon = eps_list[np.argmin(r_list)]
    nonlinear_func = func_list[np.argmin(r_list)]

    return nonlinear_func, epsilon


def find_best_eps_linear_vf(x0_data, x1_data, approx_linear_func,delta_t_list):
    """function returns epsilon with minimal mse loss for linear vector field approximation

    Parameters
    ----------
    x0_data: np.ndarray
        source data
    x1_data: np.ndarray
        target data
    approx_linear_func: int
        linear function to approximating vector field
    delta_t_list: list
        list of delta t
    Returns
    -------
    mse_list:
        list of mese loss
    a_list:
        list of A matrix
    min_delta_t:
        list of delta_t with minimal mse loss
    A:
        best A for approximating
    """
    mse_list = []
    a_list = []

    nr = x0_data.shape[0]
    for delta_t in delta_t_list:

        v_data = (x1_data - x0_data) / delta_t

        # approximate A
        _, A, _ = approx_linear_func(x0_data, v_data)

        a_list.append(A)

        eq = lambda t, x: x @ A
        x1_sol = np.zeros((nr, 2))
        for i in range(nr):
            x1_sol[i, :] = solve_ivp(eq, t_span=[0, 0.2], t_eval=[0.1], y0=x0_data[i, :]).y.reshape(-1)

        # compute mean squared error to x1
        mse = np.mean(np.sum((x1_sol - x1_data) ** 2, axis=-1))
        mse_list.append(mse)

    mse_list = np.array(mse_list)
    a_list = np.array(a_list)
    min_delta_t = delta_t_list[np.argmin(mse_list)]
    A = a_list[np.argmin(mse_list)]

    return mse_list, a_list, min_delta_t, A


def find_best_eps_nonlinear_vf(x0_data, nr_xl, delta_t, x1_data, v_data):
    """function returns epsilon with minimal mse loss for non linear vector field approximation

    Parameters
    ----------
    x0_data: np.ndarray
        source position data
    x1_data: np.ndarray
        target position data
    nr_xl: np.ndarray
        number of random selected index
    delta_t: list
        list of delta t
    v_data:
        taget vector field dataset
    Returns
    -------
    min_esp:
        best esp with minimal mse loss
    """
    min_loss = np.inf
    min_esp = 0
    eps = np.linspace(0.1, 20, 20)
    id_xl = rand_idx(x0_data, nr_xl)
    for e in eps:
        phi = get_phi(x0_data, id_xl, x0_data, e)
        C = np.linalg.lstsq(phi, v_data, rcond=1e16)[0]
        # x1_sol = np.zeros((x0_data.shape[0], 2))
        v_sol = phi @ C
        x1_sol = v_sol * delta_t + x0_data
        loss = np.mean(np.sum((x1_sol - x1_data) ** 2, axis=-1))
        if loss < min_loss:
            min_loss = loss
            min_esp = e
    return min_esp


def find_best_l_nonlinear_vf(x0_data, x1_data, v_data, delta_t, nr):
    """function returns L with minimal mse loss for non linear vector field approximation

    Parameters
    ----------
    x0_data: np.ndarray
        source position data
    x1_data: np.ndarray
        target position data
    v_data:
        taget vector field dataset
    delta_t: list
        list of delta t
    nr: int
        number of random selected index
    Returns
    -------
    idx_list:
        list of index
    mse_list:
        list of mse loss
    """
    idx_list = []
    mse_list = []
    for nr_xl in range(100, 1001, 100):

        id_xl = rand_idx(x0_data, nr_xl)
        min_esp = find_best_eps_nonlinear_vf(x0_data, nr_xl, delta_t, x1_data, v_data)
        eps = 5.33
        phi = get_phi(x0_data, id_xl, x0_data, eps)
        C = np.linalg.lstsq(phi, v_data, rcond=None)[0]
        x1_sol = np.zeros((nr, 2))
        v_sol = phi @ C

        # mse loss
        eq = lambda t, x: get_phi(x0_data, id_xl, x.reshape((1, 2)), eps) @ C
        x1_sol = np.zeros((nr, 2))
        for i in range(nr):
            x1_sol[i, :] = solve_ivp(eq, t_span=[0, 0.2], t_eval=[0.1], y0=x0_data[i, :]).y.reshape(-1)
        idx_list.append(nr_xl)
        mse = np.mean(np.sum((x1_sol - x1_data) ** 2, axis=-1))
        mse_list.append(mse)
    return idx_list, mse_list