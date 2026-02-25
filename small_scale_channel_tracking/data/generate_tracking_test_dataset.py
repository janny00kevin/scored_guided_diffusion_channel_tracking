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

from mode_selection.utils.channel_utils import calculate_coupling_matrix

# ==========================================
# 1. Configuration
# ==========================================
FREQ_GHZ = 39
TX_DIM = [7, 7]
RX_DIM = [1, 1]
R_T = 5
NUM_TEST_SAMPLES = 3000
SNR_LEVELS = [-4, -2, 0, 2, 4, 6, 8, 10]

# Tracking Physics Parameters
RHO = 0.995        # Temporal correlation (velocity dependent)
NUM_PILOTS = 5     # T >= r_T

# Input Paths
CHANNEL_FILE = os.path.join(SCRIPT_DIR, "channel", f"channel_data_SC_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_TEST_SAMPLES}samples.mat")
Z_FILE_TX = os.path.join(SCRIPT_DIR, "mode_selection", "Z_results", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_38.75GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "mode_selection", "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_38.75GHz_eigen.mat")

# Output Paths
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "testing_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"tracking_test_data_{NUM_TEST_SAMPLES}samples.pt")

def main():
    print(f"--- Generating Common Tracking Test Dataset ({FREQ_GHZ} GHz, {NUM_TEST_SAMPLES} Samples) ---")
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
        
    H_samples = np.transpose(H_complex, (2, 1, 0)) # Shape: (3000, 1, 49)
    
    # 3. Project to Modal Domain to get x_0(tau)
    print("Projecting test samples into modal domain x0(tau)...")
    H_c = np.matmul(H_samples, C_T_sqrt)
    H_tilde = np.matmul(H_c, U_T_trunc)
    x0_tau_np = np.squeeze(H_tilde) # Shape: (3000, 5)
    
    # Convert to PyTorch tensor
    x0_tau = torch.tensor(x0_tau_np, dtype=torch.complex64)
    
    # 4. Generate Ground Truth Next State: x0(tau+1)
    print(f"Simulating temporal evolution (Gauss-Markov, rho={RHO})...")
    torch.manual_seed(0) # Fix seed for reproducibility across all methods
    
    # Calculate variance for process noise Q
    x0_var = torch.var(x0_tau, dim=0) 
    Q_std = torch.sqrt((1 - RHO**2) * x0_var)
    
    # Generate process noise w
    w_real = torch.randn_like(x0_tau.real)
    w_imag = torch.randn_like(x0_tau.imag)
    w = (w_real + 1j * w_imag) / np.sqrt(2) * Q_std
    
    # State evolution
    x0_tau_plus_1 = RHO * x0_tau + w
    
    # 5. Generate Pilot Observations (Measurement Matrix M)
    print("Generating pilot measurements...")
    M_real = torch.randn(NUM_PILOTS, R_T)
    M_imag = torch.randn(NUM_PILOTS, R_T)
    M = (M_real + 1j * M_imag) / np.sqrt(2)
    
    # Clean received signal: y_clean = x0_tau_plus_1 * M^T
    y_clean = torch.matmul(x0_tau_plus_1, M.t()) # Shape: (3000, 5)
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
            "rho": RHO,
            "num_pilots": NUM_PILOTS,
            "snr_levels": SNR_LEVELS,
            "num_samples": NUM_TEST_SAMPLES,
            "process_noise_var": ((1 - RHO**2) * x0_var)
        },
        "x0_tau": x0_tau,                   # Previous state
        "x0_tau_plus_1": x0_tau_plus_1,     # Ground truth to evaluate against
        "M": M,                             # Measurement matrix
        "observations": observations        # Dict of noisy y signals
    }
    
    torch.save(dataset, OUTPUT_FILE)
    print(f"\n[Success] Independent Tracking dataset saved to:\n  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()