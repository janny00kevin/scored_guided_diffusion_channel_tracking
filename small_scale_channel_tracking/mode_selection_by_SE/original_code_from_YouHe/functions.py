import os
import numpy as np
import scipy.io as sio
import torch
from scipy.linalg import sqrtm

# Suppress potential multi-threading conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def calculate_coupling_matrix(z_mat_path, z0=50):
    """
    Converts a Z-matrix (Impedance matrix) into a Coupling Matrix (C).
    
    Args:
        z_mat_path: Path to the .mat file containing 'Z_matrix'.
        z0: Characteristic impedance (typically 50 ohms).
    Returns:
        C: Coupling matrix representing port-to-port power efficiency.
    """
    if not os.path.exists(z_mat_path):
        raise FileNotFoundError(f"File not found: {z_mat_path}")
    
    data = sio.loadmat(z_mat_path)
    if 'Z_matrix' not in data:
        raise KeyError(f"Key 'Z_matrix' not found. Available keys: {list(data.keys())}")
    
    Z = data['Z_matrix']
    N = Z.shape[0]
    
    # Calculate coupling matrix: C = 0.5 * Re{ (Z+z0I)^-1 * Z * (Z+z0I)^-H }
    term_a = Z + z0 * np.eye(N)
    x = np.linalg.solve(term_a, Z)
    y = np.linalg.solve(term_a.conj().T, np.eye(N))
    
    c = 0.5 * np.real(x @ y)
    return c

def calculate_user_rate(H_k, v_k, V_all, sigma_w2):
    """
    Calculates the achievable rate for user k considering interference.
    
    Args:
        H_k: User channel matrix (NR x NT).
        v_k: Precoding vector for user k (NT x 1).
        V_all: Matrix of all users' precoding vectors (NT x K).
        sigma_w2: Noise power.
    """
    H_k = np.array(H_k)
    v_k = np.array(v_k)
    V_all = np.array(V_all)
    NR, NT = H_k.shape
    
    # Calculate interference from other users
    interference = np.zeros((NR, NR), dtype=complex)
    for n in range(V_all.shape[1]):
        v_n = V_all[:, n:n+1]
        if not np.array_equal(v_n, v_k):
            term = H_k @ v_n @ v_n.conj().T @ H_k.conj().T
            interference += term
            
    total_noise_plus_interf = sigma_w2 * np.eye(NR) + interference
    inv_term = np.linalg.inv(total_noise_plus_interf)
    
    # SINR calculation
    core_matrix = v_k.conj().T @ H_k.conj().T @ inv_term @ H_k @ v_k
    rate = np.log2(np.linalg.det(np.eye(core_matrix.shape[0]) + core_matrix))
    
    return np.real(rate)

def generate_modal_precoder(Hc, C_T, U_T_trunct):
    """
    Generates a modal-domain precoder based on channel eigen-basis.
    """
    C_T_inv_sqrt = np.linalg.inv(sqrtm(C_T))
    h_modal = Hc @ U_T_trunct 
    _, _, vh = np.linalg.svd(h_modal, full_matrices=False)
    v_modal = vh.conj().T

    w = C_T_inv_sqrt @ U_T_trunct @ v_modal
    return w / np.linalg.norm(w, 'fro')

def generate_port_precoder(Hc, P_tx_linear=1):
    """
    Generates a full-digital port-domain precoder using SVD.
    """
    _, _, vh = np.linalg.svd(Hc, full_matrices=False)
    w_port = vh.conj().T 
    w_port = (w_port / np.linalg.norm(w_port, 'fro')) * np.sqrt(P_tx_linear)
    return w_port

def generate_hybrid_limited_precoder(Hc, C_T, U_T_trunct, P_tx_linear=1):
    """
    Generates a hybrid precoder W = Frf * E * V.
    """
    C_T_inv_sqrt = np.linalg.inv(sqrtm(C_T))
    a_ideal = C_T_inv_sqrt @ U_T_trunct
    
    # Phase shifter (analog) projection
    frf = np.exp(1j * np.angle(a_ideal)) / np.sqrt(a_ideal.shape[0])
    
    # Digital compensation matrix E (Least Squares)
    e_mat = np.linalg.pinv(frf) @ a_ideal
    
    # SVD on effective modal channel
    h_modal = Hc @ U_T_trunct 
    _, _, vh = np.linalg.svd(h_modal, full_matrices=False)
    v_modal = vh.conj().T
    
    w_hybrid = frf @ e_mat @ v_modal
    return (w_hybrid / np.linalg.norm(w_hybrid, 'fro')) * np.sqrt(P_tx_linear)

def get_V_robust(hH, B, P_tx, params):
    """
    Robust precoder optimization using Primal-Dual gradient-based approach.
    """
    hH = hH.view(1, -1) if hH.dim() == 1 else hH
    U, S, Vh = torch.linalg.svd(hH, full_matrices=False)
    
    v = Vh[0:1, :].T.conj().clone().detach().to(torch.complex128).requires_grad_(True)
    delta = torch.ones(1, params['dim_delta'], dtype=torch.complex128, requires_grad=True)
    
    beta = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    lmbda = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    
    for i in range(params['max_iter']):
        h_actual = hH + delta.view_as(hH).to(hH.dtype)
        gain = torch.norm(h_actual @ v)**2
        pwr_cons = P_tx - torch.norm(v)**2
        unc_cons = torch.real(delta @ B @ delta.conj().T).squeeze() - 1.0
        
        L = gain + beta * pwr_cons + lmbda * unc_cons
        L.backward()
        
        with torch.no_grad():
            if v.grad is not None: v += params['eta_v'] * v.grad
            if delta.grad is not None: delta -= params['eta_delta'] * delta.grad
            if beta.grad is not None: beta.copy_(torch.clamp(beta - params['eta_beta'] * pwr_cons, min=0.0))
            if lmbda.grad is not None: lmbda.copy_(torch.clamp(lmbda + params['eta_lmbda'] * unc_cons, min=0.0))
            
            for var in [v, delta, beta, lmbda]:
                if var.grad is not None: var.grad.zero_()
        
    return v.detach()

