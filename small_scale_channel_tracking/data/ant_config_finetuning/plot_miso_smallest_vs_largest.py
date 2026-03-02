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
N_T = TX_DIM[0] * TX_DIM[1] # Dynamically calculates 9 or 49

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# Dynamically set the list to use 1, 2, 3, and the full array size (N_T)
RT_LIST = [1, 2, 3, N_T]

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_1x1Rx_30000samples.mat")
# if not os.path.exists(CHANNEL_FILE):
#     CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_1x1Rx_3000samples.mat")

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
    print(f"--- MISO Capacity Analysis ({CENTER_FREQ_STR} GHz) ---")
    
    # 1. Load the Coupled System
    print("Computing Coupling Matrix...")
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_T_sqrt = sqrtm(C_T)
    
    print("Loading Eigenvectors...")
    eig_data = sio.loadmat(EIGEN_FILE_TX)
    U_T_full = eig_data['U_T_sorted']
    
    # 2. Load the Channel
    print(f"Loading Channel Data from: {os.path.basename(CHANNEL_FILE)}...")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_complex, (2, 1, 0)) 
    
    # 3. Project to the effective Physical Channel
    print("Computing Effective H_c...")
    H_c_all = np.matmul(H_samples, C_T_sqrt) 
    
    # ==========================================
    # 4. RANKING MODES BY RATE (At 10 dB SNR)
    # ==========================================
    snr_eval = 10**(10 / 10)
    single_mode_rates = np.zeros(N_T)
    
    for m in range(N_T):
        u_m = U_T_full[:, m:m+1] 
        H_tilde_m = np.matmul(H_c_all, u_m[None, :, :])
        single_mode_rates[m] = batch_capacity(H_tilde_m, snr_eval)
        
    sorted_modes = np.argsort(single_mode_rates)[::-1]
    
    print("\n" + "="*50)
    print("  EIGENVALUES ORDER ACCORDING TO ACHIEVABLE RATE")
    print("="*50)
    print("Modes sorted from Highest Rate to Lowest Rate:")
    print(sorted_modes.tolist())
    print("="*50 + "\n")
    
    # ==========================================
    # 5. SWEEP SNR FOR SMALLEST vs LARGEST
    # ==========================================
    print("Simulating Rate for Smallest and Largest RTs...")
    rates_smallest = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    rates_largest = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        for rt in RT_LIST:
            # Smallest eigenvalues 
            U_small = U_T_full[:, :rt]
            H_small = np.matmul(H_c_all, U_small[None, :, :])
            rates_smallest[rt][i] = batch_capacity(H_small, snr_linear)
            
            # Largest eigenvalues 
            U_large = U_T_full[:, -rt:]
            H_large = np.matmul(H_c_all, U_large[None, :, :])
            rates_largest[rt][i] = batch_capacity(H_large, snr_linear)

    # ==========================================
    # 6. PLOTTING (Separated into Two Figures)
    # ==========================================
    print("Generating Plots...")
    output_dir = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    # --- Figure 1: Smallest Eigenvalues ---
    plt.figure(figsize=(8, 6))
    for idx, rt in enumerate(RT_LIST):
        label_str = f'Smallest {rt} EVs' if rt != N_T else f'Full Array ({N_T} EVs)'
        plt.plot(SNR_dB_Range, rates_smallest[rt], 
                 color=colors[idx], marker=markers[idx], linestyle='-',
                 label=label_str, linewidth=2.5)
                 
    plt.title(f'Achievable Rate vs SNR (MISO {CENTER_FREQ_STR} GHz)\nSmallest Eigenvalues', fontsize=15)
    plt.xlabel('SNR [dB]', fontsize=13)
    plt.ylabel('Rate [bps/Hz]', fontsize=13)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path_small = os.path.join(output_dir, f"MISO_Rate_Smallest_EVs_{CENTER_FREQ_STR}GHz.png")
    plt.savefig(save_path_small, dpi=300)
    plt.close()
    
    # --- Figure 2: Largest Eigenvalues ---
    plt.figure(figsize=(8, 6))
    for idx, rt in enumerate(RT_LIST):
        label_str = f'Largest {rt} EVs' if rt != N_T else f'Full Array ({N_T} EVs)'
        plt.plot(SNR_dB_Range, rates_largest[rt], 
                 color=colors[idx], marker=markers[idx], linestyle='--',
                 label=label_str, linewidth=2.5)
                 
    plt.title(f'Achievable Rate vs SNR (MISO {CENTER_FREQ_STR} GHz)\nLargest Eigenvalues', fontsize=15)
    plt.xlabel('SNR [dB]', fontsize=13)
    plt.ylabel('Rate [bps/Hz]', fontsize=13)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path_large = os.path.join(output_dir, f"MISO_Rate_Largest_EVs_{CENTER_FREQ_STR}GHz.png")
    plt.savefig(save_path_large, dpi=300)
    plt.close()

    print(f"Plots successfully saved to:\n1. {save_path_small}\n2. {save_path_large}")

if __name__ == "__main__":
    main()