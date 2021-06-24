import sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def plot_time_delay(x0, delta_n):
    """
    plot time delay embedding figure in 2D

    Parameters
    ----------
    x0 : np.ndarray
        data
    delta_n : int
        step for time delay
    """
    plt.figure(figsize=(5, 5))
    points_len = x0.shape[0]
    # shift the data
    x0_t = x0[:points_len - delta_n]
    x0_t_delta_t = x0[delta_n:points_len]

    plt.scatter(x0_t, x0_t_delta_t)
    plt.xlabel("x(t)")
    plt.ylabel("x(t + \u0394 t)")
    plt.show()


def plot_time_delay_3d(x0, delta_n):
    """
    plot time delay embedding figure in 3D

    Parameters
    ----------
    x0 : np.ndarray
        data
    delta_n : int
        step for time delay
    """
    plt.figure(figsize=(5, 5))
    points_len = x0.shape[0]
    # shift the data
    x0_t = x0[:points_len - 2 * delta_n]
    x0_t_delta_t = x0[delta_n: points_len - delta_n]
    x0_t_delta_2t = x0[delta_n * 2: points_len]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection='3d')

    ax.plot(x0_t, x0_t_delta_t, x0_t_delta_2t, linewidth=0.5,
            label="time delay = {0}".format(delta_n))
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def lorenz_func(t, state, rho, sigma, beta):
    """lorenz attrator equation

    Parameters
    ----------
    t : float
        time step
    state : tuple
        current state
    rho: float
        rho in lorenz equation
    sigma: float
        sigma in lorenz equation
    """
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def plot_lorenz_burfication(y0, rho, sigma, beta, sim_time, resolution):
    """function that plot lorenz attractor phase portrait

    Parameters
    ----------
    y0_list : list
        a list contains different initial y0
    rho : float
        rho in lorenz equation
    sigma: float
        sigma in lorenz equation
    sim_time: int
        simulation time
    resolution: int
        number of points for plotting
    save: bool
        save as pdf image
    Returns
    -------
    None
    """
    t = np.linspace(0, sim_time, resolution)
    t_span = [t[0], t[-1]]
    sols = []
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection='3d')

    sol = solve_ivp(lorenz_func, t_span, y0, t_eval=t, args=(rho, sigma, beta))
    sols.append(sol)
    ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5,
            label="x = {0}, y = {1},  z = {2} ".format(y0[0], y0[1], y0[2]))
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(rf'$\sigma$ = {sigma}, $\rho$ = {round(rho, 3)}, $\beta$ = {round(beta, 3)}')
    plt.show()
    return sol
