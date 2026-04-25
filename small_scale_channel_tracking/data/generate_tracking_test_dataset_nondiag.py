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
FREQ_GHZ = 12
TX_DIM = [8, 8]
RX_DIM = [1, 1]
R_T = 64
NUM_TEST_SAMPLES = 3000
SNR_LEVELS = [-4, -2, 0, 2, 4, 6, 8, 10]

# Tracking Physics Parameters
RHO = 0.1034       # Base temporal correlation
OFF_DIAG_STD = 0.2 # Standard deviation of the off-diagonal leakage
NUM_PILOTS = R_T   # T >= r_T

# Input Paths
CHANNEL_FILE = os.path.join(SCRIPT_DIR, "channel", f"channel_data_SC_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_TEST_SAMPLES}samples.mat")
Z_FILE_TX = os.path.join(SCRIPT_DIR, "HFSS", "Z_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "HFSS", "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_eigen.mat")

# Output Paths (Updated to requested naming convention)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "training_testing_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"tracking_test_nondiag_rho{RHO:.3f}_{FREQ_GHZ}GHz_{NUM_TEST_SAMPLES}samples.pt")

def main():
    print(f"--- Generating Non-Diagonal Tracking Test Dataset ({FREQ_GHZ} GHz, {NUM_TEST_SAMPLES} Samples) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Compute Transmit Coupling Matrix and Load Eigenvectors
    print("Loading Z-matrix and computing C_T...")
    C_T = calculate_coupling_matrix(Z_FILE_TX)
    C_T_sqrt = sqrtm(C_T)
    
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
    H_tilde = np.matmul(H_c, U_T_trunc)
    x0_tau_np = np.squeeze(H_tilde) # Shape: (3000, 64)
    
    # Convert to PyTorch tensor
    x0_tau = torch.tensor(x0_tau_np, dtype=torch.complex64)
    
    # 4. Generate Ground Truth Next State: x0(tau+1) with NON-DIAGONAL A
    print(f"Simulating temporal evolution with NON-DIAGONAL A (rho={RHO}, off_diag_std={OFF_DIAG_STD})...")
    torch.manual_seed(0) # Fix seed for reproducibility across all methods
    
    # Create the non-diagonal transition matrix A
    A = torch.eye(R_T, dtype=torch.complex64) * RHO
    noise_real = torch.randn(R_T, R_T)
    noise_imag = torch.randn(R_T, R_T)
    N_off = (noise_real + 1j * noise_imag) / np.sqrt(2.0)
    
    mask = ~torch.eye(R_T, dtype=torch.bool)
    A[mask] += OFF_DIAG_STD * N_off[mask]
    
    # Spectral Stability Check
    eigenvalues = torch.linalg.eigvals(A)
    max_eig = torch.max(torch.abs(eigenvalues)).item()
    if max_eig >= 1.0:
        print(f"  Warning: Spectral radius ({max_eig:.4f}) >= 1.0. Scaling A down...")
        A = A / max_eig * 0.99
    else:
        print(f"  Matrix A is stable. Spectral radius: {max_eig:.4f}")

    # Calculate variance for process noise Q
    x0_var = torch.var(x0_tau, dim=0) 
    Q_std = torch.sqrt((1 - RHO**2) * x0_var)
    
    # Generate process noise w
    w_real = torch.randn_like(x0_tau.real)
    w_imag = torch.randn_like(x0_tau.imag)
    w = (w_real + 1j * w_imag) / np.sqrt(2) * Q_std
    
    # State evolution (Using Matrix Multiplication for the non-diagonal A)
    x0_tau_plus_1 = torch.matmul(x0_tau, A.t()) + w
    
    # 5. Generate Pilot Observations (QPSK)
    print("Generating pilot measurements...")
    M_real = (torch.randint(0, 2, size=(NUM_PILOTS, R_T)).float() * 2 - 1) 
    M_imag = (torch.randint(0, 2, size=(NUM_PILOTS, R_T)).float() * 2 - 1)
    M = (M_real + 1j * M_imag) / np.sqrt(2)
    
    # Clean received signal: y_clean = x0_tau_plus_1 * M^T
    y_clean = torch.matmul(x0_tau_plus_1, M.t()) # Shape: (3000, 64)
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
        print(f"  Generated SNR = {snr:2d} dB")
        
    # 7. Pack and Save Dataset
    dataset = {
        "config": {
            "rho": RHO, # Keep this as RHO so KF and DDIM still load it normally!
            "off_diag_std": OFF_DIAG_STD,
            "num_pilots": NUM_PILOTS,
            "snr_levels": SNR_LEVELS,
            "num_samples": NUM_TEST_SAMPLES,
            "process_noise_var": ((1 - RHO**2) * x0_var)
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