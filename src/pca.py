import numpy as np


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
    v_field_o:
        averaged vector field function
    arc_function:
        averaged sensor measurement function

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
        
    # vector field in one round (?)
    one_round_size = int(np.around((pca_set.shape[0]+delay)/7))
    v_field_o = np.zeros((one_round_size,2))
    # sensor measurement function in one round (?)
    arc_function_o = np.zeros((one_round_size,2))
    
    '''
    # interpolate for equal arc length
    v_field_interp = np.zeros(v_field.shape)
    arc_function_interp = np.zeros(arc_function.shape)
    interp = np.linspace(v_field[0,0], v_field[-1,0], pca_set.shape[0]-1)
    v_field_interp[:,1] = np.interp(interp, v_field[:,0], v_field[:,1])
    arc_function_interp[:,1] = np.interp(interp, arc_function[:,0], arc_function[:,1])
    v_field_interp[:,0] = interp
    arc_function_interp[:,0] = interp
    '''
    
    for i in range(one_round_size):
        v_field_o[i,1] = np.average(v_field[i::one_round_size,1])
        arc_function_o[i,1] = np.average(arc_function[i::one_round_size,1])
     
    v_field_o[:,1] /= v_field[one_round_size,0] / one_round_size
    
    v_field_o[:,0] = np.linspace(0, 1, one_round_size)
    arc_function_o[:,0] = np.linspace(0, 1, one_round_size)
    
    return v_field, arc_function, v_field_o, arc_function_o


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
    prediction_measurement:
        predicted measurement

    """
    
    resolution = len(v_field_o)
    # arclength calculated with vector field
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
        
    return prediction_measurement
    