import numpy as np
from src.nonlinear_approx import rand_idx, get_phi, approx_nonlinear_func
from scipy.integrate import solve_ivp

def find_best_eps(x, fx, L, e_list):
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


def find_best_eps_linear_vf(x0_data, x1_data, approx_linear_func):
    delta_t_list = np.arange(0.05, 0.15, 0.01)
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
    min_loss = np.inf
    min_esp = 0
    eps = np.linspace(0.1, 20, 20)
    id_xl = rand_idx(x0_data, nr_xl)
    for e in eps:
        phi = get_phi(x0_data, id_xl, x0_data, e)
        C = np.linalg.lstsq(phi, v_data, rcond=None)[0]
        # x1_sol = np.zeros((x0_data.shape[0], 2))
        v_sol = phi @ C
        x1_sol = v_sol * delta_t + x0_data
        loss = np.mean(np.sum((x1_sol - x1_data) ** 2, axis=-1))
        if loss < min_loss:
            min_loss = loss
            min_esp = e
    return min_esp