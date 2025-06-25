import numpy as np 

def ensure_nm_nz_shape(arr, Nm, Nz):
    """
    Ensures arr has shape (Nm, Nz) for broadcasting.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (Nm,) or (Nm, Nz) or (Nz,) or scalar.
    Nm : int
        Number of halo mass bins.
    Nz : int
        Number of redshift bins.

    Returns
    -------
    arr : np.ndarray
        Array reshaped to (Nm, Nz).
    """

    if arr.shape == (Nm,):
        arr = arr[:,np.newaxis]  # (Nm, 1)
    return arr