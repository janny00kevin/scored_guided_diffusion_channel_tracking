import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm
import torch
import h5py
import os
import sys

# Add project root to path for utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..")) 

from utils.channel_utils import calculate_coupling_matrix

# ==========================================
# 1. Configuration
# ==========================================
FREQ_GHZ = 38
TX_DIM = [8, 8]
RX_DIM = [1, 1]
R_T = 54
NUM_TEST_SAMPLES = 3000
SNR_LEVELS = [-4, -2, 0, 2, 4, 6, 8, 10]

# Tracking Physics Parameters
if FREQ_GHZ == 12: RHO = 0.1034     # Base temporal correlation
elif FREQ_GHZ == 38: RHO = 0.0001
OFF_DIAG_STD = 0.2 # Standard deviation of the off-diagonal leakage
NUM_PILOTS = 64*1   # T >= r_T
MODE = 'modal' # 'spatial' or 'modal'

# Input Paths
CHANNEL_FILE = os.path.join(SCRIPT_DIR, "channel", f"channel_data_SC_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_TEST_SAMPLES}samples.mat")
Z_FILE_TX = os.path.join(SCRIPT_DIR, "HFSS", "Z_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "HFSS", "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_eigen.mat")

# Output Paths (Updated to requested naming convention)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "training_testing_dataset")
if MODE == 'spatial':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 
                               f"tracking_test_nondiag_spatial_T{NUM_PILOTS}_rho{RHO:.3f}_{FREQ_GHZ}GHz_{NUM_TEST_SAMPLES}samples.pt")
elif MODE == 'modal':
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 
                               f"tracking_test_nondiag_rT{R_T}_T{NUM_PILOTS}_rho{RHO:.3f}_{FREQ_GHZ}GHz_{NUM_TEST_SAMPLES}samples.pt")

def main():
    print(f"--- Generating Non-Diagonal Tracking Test Dataset ({FREQ_GHZ} GHz, {NUM_TEST_SAMPLES} Samples) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Compute Transmit Coupling Matrix and Load Eigenvectors
    print("Loading Z-matrix and computing C_T...")
    C_T = calculate_coupling_matrix(Z_FILE_TX)
    C_T_sqrt = sqrtm(C_T)
    
    if MODE == 'modal':
        print("Loading modal eigenvectors...")
        U_T_full = sio.loadmat(EIGEN_FILE_TX)['U_T_sorted']
        U_T_trunc = U_T_full[:, :R_T]
    
    # 2. Load Raw Test Channel Samples
    print(f"Loading raw test channel samples from:\n  {os.path.basename(CHANNEL_FILE)}")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
        
    H_samples = np.transpose(H_complex, (2, 1, 0)) # Shape: (3000, 1, 64)
    
    # 3. Project to Modal Domain to get x_0(tau)
    print("Projecting test samples into modal domain x0(tau)...")
    H_c = np.matmul(H_samples, C_T_sqrt)
    # x0_tau_np = np.squeeze(H_c)
    if MODE == 'spatial':
        x0_tau_np = np.squeeze(H_c) # Shape: (3000, NUM_PILOTS)
    elif MODE == 'modal':
        H_tilde = np.matmul(H_c, U_T_trunc)
        x0_tau_np = np.squeeze(H_tilde) # Shape: (3000, r_T)
    
    # Convert to PyTorch tensor
    x0_tau = torch.tensor(x0_tau_np, dtype=torch.complex64)
    
    # 4. Generate Ground Truth Next State: x0(tau+1) with NON-DIAGONAL A
    print(f"Simulating temporal evolution with NON-DIAGONAL A (rho={RHO}, off_diag_std={OFF_DIAG_STD})...")
    torch.manual_seed(0) # Fix seed for reproducibility across all methods
    
    # Create the non-diagonal transition matrix A
    if MODE == 'spatial':
        dim = TX_DIM[0] * TX_DIM[1] # 64
    elif MODE == 'modal':
        dim = R_T 
    A = torch.eye(dim, dtype=torch.complex64) * RHO
    noise_real = torch.randn(dim, dim)
    noise_imag = torch.randn(dim, dim)
    N_off = (noise_real + 1j * noise_imag) / np.sqrt(2.0)
    
    mask = ~torch.eye(dim, dtype=torch.bool)
    A[mask] += OFF_DIAG_STD * N_off[mask]
    
    # Spectral Stability Check for \A
    eigenvalues = torch.linalg.eigvals(A)
    max_eig = torch.max(torch.abs(eigenvalues)).item()
    if max_eig >= 1.0:
        print(f"  Warning: Spectral radius ({max_eig:.4f}) >= 1.0. Scaling A down...")
        A = A / max_eig * 0.99
    else:
        print(f"  Matrix A is stable. Spectral radius: {max_eig:.4f}")

    # # Always use spatial dimensions for evolution
    # N_T = TX_DIM[0] * TX_DIM[1] # 64
    
    # Calculate variance for spatial process noise Q
    H_c_tensor = torch.squeeze(torch.tensor(H_c, dtype=torch.complex64))
    spatial_var = torch.var(H_c_tensor, dim=0)  ################
    # spatial_var = torch.ones(N_T)     
    # Q_std_spatial = torch.sqrt((1 - RHO**2) * spatial_var)  
    
    # # Generate spatial process noise w
    # w_real = torch.randn(NUM_TEST_SAMPLES, N_T)
    # w_imag = torch.randn(NUM_TEST_SAMPLES, N_T)
    # w_spatial = (w_real + 1j * w_imag) / np.sqrt(2) * Q_std_spatial
    
    # Evolve the spatial channel first
    # No process noise added. The uncertainty is strictly the spatial leakage inside A.
    # Hc_tau_plus_1 = torch.matmul(H_c_tensor, A.t())
    
    if MODE == 'spatial':
        x0_tau_plus_1 = torch.matmul(H_c_tensor, A.t())
    elif MODE == 'modal':
        # Project the evolved spatial channel to modal domain
        U_T_tensor = torch.tensor(U_T_trunc, dtype=torch.complex64)
        x0_tau_plus_1 = torch.matmul(torch.matmul(H_c_tensor, U_T_tensor), A.t())
    # x0_tau_plus_1 = torch.matmul(x0_tau, A.t()) + w
    
# 5. Generate Pilot Observations (QPSK)
    print("Generating pilot measurements...")
    
    # 5a. Always generate the QPSK pilots in the tracker's native domain
    if MODE == 'spatial':
        dim = TX_DIM[0] * TX_DIM[1] # 64
    elif MODE == 'modal':
        dim = R_T # 38
        
    M_real = (torch.randint(0, 2, size=(NUM_PILOTS, dim)).float() * 2 - 1) 
    M_imag = (torch.randint(0, 2, size=(NUM_PILOTS, dim)).float() * 2 - 1)
    M = (M_real + 1j * M_imag) / np.sqrt(2)
    
    # if MODE == 'spatial':
    #     # Standard spatial transmission
    #     M_tracker = M_base 
    #     y_clean = torch.matmul(x0_tau_plus_1, M_tracker.t())
        
    # elif MODE == 'modal':
    #     # Math: M_spatial = M_modal * U_{T, r_T}^T
    #     U_T_tensor = torch.tensor(U_T_trunc, dtype=torch.complex64)
    #     M_spatial = torch.matmul(M_base, U_T_tensor.t()) # Shape: (NUM_PILOTS, 64)
        
    #     # Pass the physically precoded pilots through the TRUE SPATIAL channel (Hc_tau_plus_1)
    #     # (3000, 64) @ (64, NUM_PILOTS) 
    #     y_clean = torch.matmul(Hc_tau_plus_1, M_spatial.t()) 
        
    #     # The tracker only needs the modal pilots, it doesn't need to know about U_T
    #     M_tracker = M_base
    
    y_clean = torch.matmul(x0_tau_plus_1, M.t())
    
    sig_power = torch.mean(torch.abs(y_clean)**2)
    
    # 6. Generate Noisy Observations for each SNR
    observations = {}
    print("Adding AWGN for SNR levels...")
    for snr in SNR_LEVELS:
        sigma_n2 = sig_power * (10 ** (-snr / 10.0))
        noise_real = torch.randn_like(y_clean.real)
        noise_imag = torch.randn_like(y_clean.imag)
        noise = (noise_real + 1j * noise_imag) / np.sqrt(2) * torch.sqrt(sigma_n2)
        
        y_obs = y_clean + noise
        observations[snr] = y_obs
        # print(f"  Generated SNR = {snr:2d} dB")
        
    # 7. Pack and Save Dataset
    dataset = {
        "config": {
            "rho": RHO, # Keep this as RHO so KF and DDIM still load it normally!
            "off_diag_std": OFF_DIAG_STD,
            "num_pilots": NUM_PILOTS,
            "snr_levels": SNR_LEVELS,
            "num_samples": NUM_TEST_SAMPLES,
            "process_noise_var": ((1 - RHO**2) * spatial_var)
        },
        "true_A_matrix": A,                 # Storing true A just for reference
        "x0_tau": x0_tau,                   # Previous state
        "x0_tau_plus_1": x0_tau_plus_1,     # Ground truth to evaluate against
        "M": M,                             # Measurement matrix
        "observations": observations        # Dict of noisy y signals
    }
    
    torch.save(dataset, OUTPUT_FILE)
    print(f"\n[Success] Non-Diagonal Tracking dataset saved to:\n  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()