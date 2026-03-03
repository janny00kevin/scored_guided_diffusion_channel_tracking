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

TX_DIM = [7, 7] # Change to [7, 7] when ready for the big array
RX_DIM = [1, 1]
N_T = TX_DIM[0] * TX_DIM[1] 

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_1x1Rx_30000samples.mat")
if not os.path.exists(CHANNEL_FILE):
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
    print(f"--- MISO Capacity Analysis: EVs < 1 vs Full Array ({CENTER_FREQ_STR} GHz) ---")
    
    # 1. Load the Coupled System
    print("Computing Coupling Matrix...")
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_T_sqrt = sqrtm(C_T)
    
    print("Loading Eigenvectors and Eigenvalues...")
    eig_data = sio.loadmat(EIGEN_FILE_TX)
    U_T_full = eig_data['U_T_sorted']
    
# 2. Extract Eigenvalues and find indices where |EV| < 1
    # Using the exact key confirmed from the .mat file
    eig_vals = np.abs(eig_data['lambda_sorted'].flatten())
            
    # if eig_vals is None:
    #     print("\nError: Could not find the eigenvalues in your .mat file.")
    #     print("Available variables in file:", [k for k in eig_data.keys() if not k.startswith('__')])
    #     print("Please ensure your MATLAB script saves the eigenvalues (e.g., 'lambda_T_sorted').")
    #     return
        
    valid_indices = np.where(eig_vals < 1.0)[0]
    K_valid = len(valid_indices)
    
    print(f"Total Modes: {N_T}")
    print(f"Modes with |EV| < 1: {K_valid}")
    print(f"Indices used: {valid_indices.tolist()}")
    
    if K_valid == 0:
        print("Warning: No eigenvalues are less than 1. Cannot plot the filtered curve.")
        return
        
    # Extract only the efficient eigenvectors
    U_efficient = U_T_full[:, valid_indices]
    
    # 3. Load the Channel
    print(f"Loading Channel Data from: {os.path.basename(CHANNEL_FILE)}...")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_complex, (2, 1, 0)) 
    
    # 4. Project to the effective Physical Channel
    print("Computing Effective H_c...")
    H_c_all = np.matmul(H_samples, C_T_sqrt) 
    
    # ==========================================
    # 5. SWEEP SNR 
    # ==========================================
    print("Simulating Rates...")
    rates_efficient = np.zeros(len(SNR_dB_Range))
    rates_full = np.zeros(len(SNR_dB_Range))
    
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        # Rate for modes with EV < 1
        H_eff = np.matmul(H_c_all, U_efficient[None, :, :])
        rates_efficient[i] = batch_capacity(H_eff, snr_linear)
        
        # Rate for all modes (Full Array)
        H_full = np.matmul(H_c_all, U_T_full[None, :, :])
        rates_full[i] = batch_capacity(H_full, snr_linear)

    # ==========================================
    # 6. PLOTTING
    # ==========================================
    print("Generating Plot...")
    plt.figure(figsize=(8, 6))
    
    # Plot Efficient Modes
    plt.plot(SNR_dB_Range, rates_efficient, 
             color='#1f77b4', marker='o', linestyle='-',
             label=f'|$\lambda$| < 1: {K_valid} Modes', linewidth=2.5)
             
    # Plot Full Array
    plt.plot(SNR_dB_Range, rates_full, 
             color='#d62728', marker='s', linestyle='--',
             label=f'Full {N_T} Modes', linewidth=2.5)
             
    plt.title(f'Achievable Rate vs SNR (MISO {CENTER_FREQ_STR} GHz)\nEfficient Modes vs Full Array', fontsize=15)
    plt.xlabel('SNR [dB]', fontsize=13)
    plt.ylabel('Rate [bps/Hz]', fontsize=13)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"MISO_Rate_EV_less_than_1_vs_Full_{CENTER_FREQ_STR}GHz.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to:\n{save_path}")

if __name__ == "__main__":
    main()