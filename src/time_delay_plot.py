import sys
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def plot_time_delay(x0, delta_n):
    plt.figure(figsize=(10, 10))
    points_len = x0.shape[0]
    # shift the data
    x0_t = x0[:points_len - delta_n]
    x0_t_delta_t = x0[delta_n:points_len]
    
    plt.scatter(x0_t, x0_t_delta_t)
    plt.xlabel("x(t)")
    plt.ylabel("x(t + \u0394 t)")
    plt.show()