import numpy as np
import scipy.io as sio
import os

def calculate_coupling_matrix(z_mat_path, z0=50):
    if not os.path.exists(z_mat_path): 
        raise FileNotFoundError(f"Not found: {z_mat_path}")
    data = sio.loadmat(z_mat_path)
    Z = data['Z_matrix']
    N = Z.shape[0]
    Term_A = Z + z0 * np.eye(N)
    X = np.linalg.solve(Term_A, Z)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N))
    return 0.5 * np.real(X @ Y)

def load_eigenvectors(eigen_path):
    if not os.path.exists(eigen_path): 
        raise FileNotFoundError(f"Not found: {eigen_path}")
    return sio.loadmat(eigen_path)['U_T_sorted']

def batch_capacity(H_batch, snr_linear):
    """
    Vectorized Shannon Capacity calculation for an array of matrices.
    H_batch shape: (samples, N_Rx, N_Tx)
    """
    N_Rx = H_batch.shape[1]
    I = np.eye(N_Rx, dtype=complex)[None, :, :]
    H_batch_H = H_batch.conj().transpose(0, 2, 1)
    inner = I + snr_linear * (H_batch @ H_batch_H)
    
    # slogdet returns (sign, log_determinant). We only need log_determinant.
    _, logdet = np.linalg.slogdet(inner) 
    return np.mean(logdet) / np.log(2)