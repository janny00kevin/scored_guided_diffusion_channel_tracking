import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm, solve, norm
import os
import h5py

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FREQ_GHZ = 39
ASF = 2  # Antenna Spacing Factor

# Dimensions
TX_DIM = [7, 7]  # [rows, cols]
RX_DIM = [2, 2]
N_T = TX_DIM[0] * TX_DIM[1] # 49
N_R = RX_DIM[0] * RX_DIM[1] # 4

# Rank Truncation (from your prompt)
R_T = 6
R_R = 4

# Characteristic Impedance
Z0 = 50 

# Define paths relative to the script's location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
Z_RESULTS_DIR = os.path.join(DATA_DIR, "impedance_matrix_matlab", "Z_results")
EIGEN_RESULTS_DIR = os.path.join(DATA_DIR, "impedance_matrix_matlab", "eigen_result")
CHANNEL_DIR = os.path.join(DATA_DIR, "channel")

# Dynamic Filename (using the fixed format)
CHANNEL_FILE = os.path.join(CHANNEL_DIR, f"channel_data_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_3000samples.mat")

Z_FILE_TX = os.path.join(Z_RESULTS_DIR, f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_Z.mat")
Z_FILE_RX = os.path.join(Z_RESULTS_DIR, f"{RX_DIM[0]}x{RX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_Z.mat")

EIGEN_FILE_TX = os.path.join(EIGEN_RESULTS_DIR, f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_eigen.mat")
EIGEN_FILE_RX = os.path.join(EIGEN_RESULTS_DIR, f"{RX_DIM[0]}x{RX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_eigen.mat")

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(z_mat_path, z0=50):
    """
    Replicates Calculate_Coupling_Matrix.m logic:
    C_T = 0.5 * real( (Z_T + Z0*I)^-1 * Z_T * (Z_T + Z0*I)^-H )
    """
    if not os.path.exists(z_mat_path):
        raise FileNotFoundError(f"Z-Matrix file not found: {z_mat_path}")
        
    data = sio.loadmat(z_mat_path)
    Z = data['Z_matrix']
    N = Z.shape[0]
    I = np.eye(N)
    
    # Term A = Z_T + Z0 * I
    Term_A = Z + z0 * I
    
    # Inner Product = (Term_A \ Z_T) / (Term_A')
    # Equivalent to: inv(Term_A) @ Z_T @ inv(Term_A.conj().T)
    
    # 1. Solve Term_A * X = Z_T  => X = inv(Term_A) * Z_T
    X = solve(Term_A, Z)
    
    # 2. Multiply by inv(Term_A^H)
    # We want X @ inv(Term_A.H). 
    # Let Y = inv(Term_A.H). We solve Term_A.H * Y = I
    Term_A_H = Term_A.conj().T
    Y = solve(Term_A_H, np.eye(N))
    
    Inner_Product = X @ Y
    
    C_matrix = 0.5 * np.real(Inner_Product)
    return C_matrix

def load_sorted_eigenvectors(eigen_path):
    """Loads U_T_sorted from the .mat file."""
    if not os.path.exists(eigen_path):
        raise FileNotFoundError(f"Eigen result file not found: {eigen_path}")
    
    data = sio.loadmat(eigen_path)
    # File contains 'U_T_sorted' and 'lambda_sorted'
    U_sorted = data['U_T_sorted']
    return U_sorted

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"--- Starting Modal Domain Validation ({FREQ_GHZ} GHz) ---")
    
    # 1. Load Channel Data
    # Shape is (3000, N_Rx, N_Tx)
    print(f"Loading Channel Data: {CHANNEL_FILE}")
    if not os.path.exists(CHANNEL_FILE):
        print(f"Warning: Channel file not found. Please ensure generating channel data for {FREQ_GHZ}GHz first.")
        return
        
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_samples_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_samples_complex, (2, 1, 0))
    num_samples = H_samples.shape[0]
    print(f"Loaded {num_samples} samples. Shape: {H_samples.shape}")

    # 2. Compute Coupling Matrices (C_T, C_R)
    print("Computing Coupling Matrices...")
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_R = calculate_coupling_matrix(Z_FILE_RX, Z0)
    
    # Compute C^(1/2)
    C_T_sqrt = sqrtm(C_T)
    C_R_sqrt = sqrtm(C_R)
    
    # 3. Load Eigenvectors
    print("Loading Sorted Eigenvectors...")
    U_T_full = load_sorted_eigenvectors(EIGEN_FILE_TX) # 49x49
    U_R_full = load_sorted_eigenvectors(EIGEN_FILE_RX) # 4x4
    
    # 4. Truncate Eigenvectors (Modal Subspaces)
    # U_{T, rT} (49 x 6)
    U_T_trunc = U_T_full[:, :R_T]
    # U_{R, rR} (4 x 4)
    U_R_trunc = U_R_full[:, :R_R]
    
    print(f"Truncation: Tx Rank {R_T}/{N_T}, Rx Rank {R_R}/{N_R}")

    # Factoring C_T_sqrt approx U_T_trunc @ Gamma_T_sqrt @ U_T_trunc^H
    # Projection: Gamma_sqrt = U^H @ C_sqrt @ U
    Gamma_T_sqrt = U_T_trunc.conj().T @ C_T_sqrt @ U_T_trunc
    Gamma_R_sqrt = U_R_trunc.conj().T @ C_R_sqrt @ U_R_trunc

    # 5. Define Test Vector s (Modal Domain Excitation)
    # Equation: s can equal [1 0 ... 0]^T
    s = np.zeros((R_T, 1))
    s[0, 0] = 1 
    
    # Compute Excitation Vector x (Spatial Domain)
    # Equation: x = U_{T, rT} * s
    x = U_T_trunc @ s
    
    # 6. Loop over samples
    error_list = []
    
    print("Running Input-Output Validation Loop...")
    
    for i in range(num_samples):
        # Raw Ray-Tracing Channel (N_R x N_T)
        H_spatial = H_samples[i, :, :] 
        
        # --- A. Full Channel (Physics-based Simulation) ---
        # Equation: y_full = C_R^1/2 * H * C_T^1/2 * x
        # Note: H_spatial from chan_gen is essentially the "propagation" H. 
        # The physical voltages account for coupling at both ends.
        
        # Effective System Channel H_sys
        H_sys = C_R_sqrt @ H_spatial @ C_T_sqrt
        
        # Received spatial signal
        y_full = H_sys @ x
        
        # --- B. Modal Channel Construction (Approximation) ---
        # H_c = U_R @ Gamma_R @ U_R^H @ H_spatial @ U_T @ Gamma_T @ U_T^H
        # This represents the channel confined to the truncated modal subspace
        H_c = (U_R_trunc @ Gamma_R_sqrt @ U_R_trunc.conj().T @ 
               H_spatial @ 
               U_T_trunc @ Gamma_T_sqrt @ U_T_trunc.conj().T)
        
        # H_tilde_c = U_R^H @ H_c @ U_T
        H_tilde_c = U_R_trunc.conj().T @ H_c @ U_T_trunc
        
        # 2. Compute received modal signal
        # Equation: y_modal = U_{R,rR} * H_tilde_c * s
        # Wait, strictly speaking y_modal in the figure is defined as:
        # y_modal = U_{R,rR} * H_tilde_c * s
        # However, looking at the error equation: || U^H y_full - y_modal ||
        # If y_modal = H_tilde_c * s (purely modal domain vector, size rR x 1)
        # Then U^H * y_full (size rR x 1) matches dimensions.
        
        # Let's align exactly with the error equation image:
        # y_modal in the text says "in C^{rR}", which implies it's the coefficient vector.
        # So y_modal = H_tilde_c * s
        y_modal = H_tilde_c @ s
        
        # --- C. Compute Normalized Error ---
        # Numerator: || U_{R,rR}^H * y_full - y_modal ||_2
        # Denominator: || U_{R,rR}^H * y_full ||_2
        
        projected_y_full = U_R_trunc.conj().T @ y_full
        
        numerator = norm(projected_y_full - y_modal, 2)
        denominator = norm(projected_y_full, 2)
        
        if denominator < 1e-12:
            current_error = 0.0
        else:
            current_error = numerator / denominator
            
        error_list.append(current_error)

    # 7. Results
    mean_error = np.mean(error_list)
    print("\n" + "="*40)
    print(f"VALIDATION RESULTS ({num_samples} samples)")
    print("="*40)
    print(f"Frequency: {FREQ_GHZ} GHz")
    print(f"Tx Array:  {TX_DIM} (Rank {R_T} used)")
    print(f"Rx Array:  {RX_DIM} (Rank {R_R} used)")
    print("-" * 40)
    print(f"Mean Normalized Error: {mean_error:.6f}")
    print(f"Min Error: {np.min(error_list):.6f}")
    print(f"Max Error: {np.max(error_list):.6f}")
    print("="*40)

if __name__ == "__main__":
    main()