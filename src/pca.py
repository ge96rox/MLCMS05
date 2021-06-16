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