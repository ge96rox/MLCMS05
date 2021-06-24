import numpy as np
from scipy.spatial.distance import cdist


def pca(dataset, num_pc=-1):
    """function that plot fire evac dataset

    Parameters
    ----------
    dataset : np.ndarray, shape [N, D]
        data of distributed people within the MI building
    num_pc: int
        number of principle components
    Returns
    -------
    u:
        u matrix from SVD
    s:
        s matrix from SVD
    vh:
        v matrix from SVD
    s_truncated:
         truncate s matrix

    """
    # L
    if (num_pc == -1):
        num_pc = min(dataset.shape)

        # perform data centering
    pca_mean = [np.mean(dataset.T[i]) for i in range(dataset.shape[1])]
    pca_centered = np.array([dataset[i] - pca_mean for i in range(dataset.shape[0])])

    # perform svd
    u, s_values, vh = np.linalg.svd(pca_centered)

    # construct sigma matrix
    s_fullrank = np.diag(s_values)
    s = np.zeros((u.shape[1], vh.shape[0]))
    s[:s_fullrank.shape[0], :s_fullrank.shape[1]] = s_fullrank

    # truncate sigma according to L by setting 0
    s_truncated = s.copy()
    s_truncated[num_pc:, ] = 0

    return u, s, vh, s_truncated


def arc_length_velocity(pca_set, measurement_set, dim_index, delay):
    
    """compute two functions arclength velocity against arclength and sensor measurement against arclength

    Parameters
    ----------
    pca_set : np.ndarray
        set of points after PCA
    measurement_set : np.ndarray
        origin data from sensors
    dim_index: int
        index of sensor of interest
    delay: int
        size of sliding window in time-delay embedding
    Returns
    -------
    v_field:
        vector field function
    arc_function:
        sensor measurement function

    """
    
    # vektor field
    v_field = np.zeros((pca_set.shape[0]-1, 2))
    # sensor measurement against arc length
    arc_function = np.zeros((pca_set.shape[0]-1, 2))
    list_length = []
    
    for i in range(pca_set.shape[0]-1):
        length = np.linalg.norm(pca_set[i]-pca_set[i+1])
        list_length.append(length)
        v_field[i,0] = sum(list_length)
        v_field[i,1] = length
        arc_function[i,0] = v_field[i,0]
        arc_function[i,1] = measurement_set[delay+i,dim_index]
          
    return v_field, arc_function


def rbf_sin(x, fx, L, e, p):
    
    def rbf(x, x_l, eps, p):
        return np.exp(-np.abs(np.cos(cdist(x,x_l)/p))/eps)
    
    def rand_idx(x, nr_randpts):
        return np.random.permutation(x.shape[0])[0:nr_randpts]
    
    def get_phi(x0_data, id_xl, current_x_data, eps, p):
        phi = rbf(current_x_data, x0_data[id_xl], eps, p)
        return phi
    
    def approximated_func(x_new):
        dist_new = cdist(x_new, x[id_xl])
        phi_new = np.exp(-np.abs(np.cos(dist_new/p))/epsilon)
        return phi_new @ C
    
    # choose L random elements of the data
    id_xl = rand_idx(x, L)

    # choose epsilon similar to diffusion map
    dist = cdist(x, x[id_xl])
    epsilon = e * np.max(dist)
    phi = get_phi(x, id_xl, x, epsilon, p)

    C, res, _, _ = np.linalg.lstsq(phi, fx, rcond=1e-16)
    
    return approximated_func, C, res, epsilon


def predict(num_days, v_field_o, arc_function_o):
    """predict future measurement

    Parameters
    ----------
    num_days: int
        number of days to predict
    v_field_o: np.ndarray
        averaged vector field function
    arc_function_o: np.ndarray
        averaged sensor measurement function
    Returns
    -------
    prediction_arc:
        predicted arc length
    prediction_measurement:
        predicted measurement
    """ 
    # arclength calculated with vector field
    resolution = len(v_field_o)
    prediction_arc = np.zeros((resolution*num_days, 2))
    # predicted measurement
    prediction_measurement = np.zeros((resolution*num_days, 2))
    
    # extend matrix according to number of days
    # time axis of measurement function should be taken care of
    v_field_tile = np.tile(v_field_o, (num_days,1))
    arc_function_tile = np.tile(arc_function_o, (num_days,1))
    arc_function_tile[:,0] = np.linspace(0, num_days, resolution*num_days, endpoint=False)
    
    prediction_arc[:,0] = np.linspace(0, num_days, resolution*num_days, endpoint=False)
    # differential equation calculation with euler methods
    for i in range(resolution*num_days-1):
        prediction_arc[i+1,1] = prediction_arc[i,1] + v_field_tile[i,1] / resolution
    
    prediction_measurement[:,0] = np.linspace(0, num_days, resolution*num_days, endpoint=False)
    # find the value in measurement function according to the arclength
    prediction_measurement[:,1] = np.interp(prediction_arc[:,1], arc_function_tile[:,0], arc_function_tile[:,1])
        
    return prediction_arc, prediction_measurement


def rbf_predict(v_field, arc_function):

    predict = np.zeros((28000,1))
    predict_measurement = np.zeros((28000,1))
    
    for i in range(28000-1):
        predict[i+1,0] = predict[i,0] + v_field(predict[i,0].reshape((-1,1)))
        
    predict_measurement = arc_function(predict[:,0].reshape((-1,1)))
    
    return predict_measurement