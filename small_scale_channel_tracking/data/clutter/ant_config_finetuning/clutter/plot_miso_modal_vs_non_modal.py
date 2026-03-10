import numpy as np
from scipy.linalg import sqrtm
import scipy.io as sio
import os
import h5py
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
CENTER_FREQ_STR = "38.75"  
CHANNEL_FREQ_GHZ = 39

TX_DIM = [7, 7] 
RX_DIM = [1, 1]
N_T = TX_DIM[0] * TX_DIM[1] 

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_1x1Rx_3000samples.mat")

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(z_mat_path, z0=50):
    if not os.path.exists(z_mat_path):
        raise FileNotFoundError(f"Z-matrix file not found: {z_mat_path}")
    data = sio.loadmat(z_mat_path)
    Z = data['Z_matrix']
    N = Z.shape[0]
    Term_A = Z + z0 * np.eye(N)
    X = np.linalg.solve(Term_A, Z)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N))
    return 0.5 * np.real(X @ Y)

def batch_capacity(H_batch, snr_linear):
    """ Vectorized Shannon Capacity: E[log2(det(I + SNR * H*H^H))] """
    N_Rx = H_batch.shape[1]
    I = np.eye(N_Rx, dtype=complex)[None, :, :]
    H_batch_H = H_batch.conj().transpose(0, 2, 1)
    inner = I + snr_linear * (H_batch @ H_batch_H)
    _, logdet = np.linalg.slogdet(inner) 
    return np.mean(logdet) / np.log(2)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"--- MISO Capacity Analysis: Efficient Modes vs Full Array vs Pure H ({CENTER_FREQ_STR} GHz) ---")
    
    # 1. Load the Coupled System Data
    print("Computing Coupling Matrix...")
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_T_sqrt = sqrtm(C_T)
    
    print("Loading Eigenvectors and Eigenvalues...")
    eig_data = sio.loadmat(EIGEN_FILE_TX)
    U_T_full = eig_data['U_T_sorted']
    
    # Extract Eigenvalues and find indices where |EV| < 1
    eig_vals = np.abs(eig_data['lambda_sorted'].flatten())
    valid_indices = np.where(eig_vals < 1.0)[0]
    K_valid = len(valid_indices)
    
    print(f"Total Modes: {N_T}")
    print(f"Modes with |EV| < 1: {K_valid}")
    
    U_efficient = U_T_full[:, valid_indices]
    
    # 2. Load the Channel
    print(f"Loading Channel Data: {os.path.basename(CHANNEL_FILE)}...")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_complex, (2, 1, 0)) # (Samples, 1, 49)
    
    # 3. Project to the effective Physical Coupled Channel
    print("Computing Effective H_c (with Coupling)...")
    H_c_all = np.matmul(H_samples, C_T_sqrt) 
    
    # ==========================================
    # 4. SWEEP SNR 
    # ==========================================
    print("Simulating Rates...")
    rates_efficient = np.zeros(len(SNR_dB_Range))
    rates_full = np.zeros(len(SNR_dB_Range))
    rates_pure_h = np.zeros(len(SNR_dB_Range))
    
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        # A. Rate for efficient modes (|lambda| < 1)
        H_eff = np.matmul(H_c_all, U_efficient[None, :, :])
        rates_efficient[i] = batch_capacity(H_eff, snr_linear)
        
        # B. Rate for full coupled array (All 49 Modes)
        H_full = np.matmul(H_c_all, U_T_full[None, :, :])
        rates_full[i] = batch_capacity(H_full, snr_linear)
        
        # C. Rate for Pure H (No Coupling / No Eigenvector Projection)
        # Directly use H_samples without C_T_sqrt or U_T
        rates_pure_h[i] = batch_capacity(H_samples, snr_linear)

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    print("Generating Plot...")
    plt.figure(figsize=(9, 6))
    
    # Plot Pure H (No eigenvectors/No Coupling)
    plt.plot(SNR_dB_Range, rates_pure_h, 
             color='#2ca02c', marker='^', linestyle=':',
             label='Pure Channel (No Coupling/Modes)', linewidth=2.0)
    
    # Plot Efficient Modes
    plt.plot(SNR_dB_Range, rates_efficient, 
             color='#1f77b4', marker='o', linestyle='-',
             label=f'Efficient Modes ($|\\lambda| < 1$) [{K_valid} Modes]', linewidth=2.5)
             
    # Plot Full Coupled Array
    plt.plot(SNR_dB_Range, rates_full, 
             color='#d62728', marker='s', linestyle='--',
             label=f'Full Array with Coupling ({N_T} Modes)', linewidth=2.5)
             
    plt.title(f'MISO Achievable Rate Comparison ({CENTER_FREQ_STR} GHz)\nCoupling-Aware vs. Pure Channel', fontsize=14)
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Rate [bps/Hz]', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"MISO_Rate_Comparison_PureH_vs_Modes_{CENTER_FREQ_STR}GHz.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()