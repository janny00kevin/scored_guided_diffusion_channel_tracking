import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm
import os
import h5py
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

# --- Mode Selection ---
# Set to 39 for 39 GHz band, or 2 for 2 GHz band
MODE = 2

if MODE == 39:
    CENTER_FREQ_STR = "38.75"  # Exact string used in your new MATLAB saves
    CHANNEL_FREQ_GHZ = 39      # Integer used for channel tracing filename
    RT_LIST = [1, 2, 3, 49]        # The rank truncations to test
elif MODE == 2:
    CENTER_FREQ_STR = "2.06"
    CHANNEL_FREQ_GHZ = 2
    RT_LIST = [10, 20, 43, 49]     # Example list for 2 GHz

# Array Dimensions
TX_DIM = [7, 7]
RX_DIM = [2, 2]
N_T = TX_DIM[0] * TX_DIM[1] # 49
N_R = RX_DIM[0] * RX_DIM[1] # 4

R_R = 4  # Keep all receiver modes

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Define Paths ---
# Tx and Rx files are now both pulled from the new 'mode_tracking' folder
Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"7x7_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"7x7_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

Z_FILE_RX = os.path.join(SCRIPT_DIR, "Z_results", f"2x2_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_RX = os.path.join(SCRIPT_DIR, "eigen_result", f"2x2_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

# Channel File (Always uses the rounded frequency format from channel_gen.m)
CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_{CHANNEL_FREQ_GHZ}GHz_7x7Tx_2x2Rx_3000samples.mat")

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(z_mat_path, z0=50):
    if not os.path.exists(z_mat_path): 
        raise FileNotFoundError(f"Not found: {z_mat_path}")
    data = sio.loadmat(z_mat_path)
    Z = data['Z_matrix']
    N = Z.shape[0]
    Term_A = Z + z0 * np.eye(N)
    # C = 0.5 * real( inv(A) * Z * inv(A^H) )
    X = np.linalg.solve(Term_A, Z)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N))
    return 0.5 * np.real(X @ Y)

def load_eigenvectors(eigen_path):
    if not os.path.exists(eigen_path): 
        raise FileNotFoundError(f"Not found: {eigen_path}")
    return sio.loadmat(eigen_path)['U_T_sorted']

def capacity_logdet(H, snr_linear):
    N_Rx, N_Tx = H.shape
    Identity = np.eye(N_Rx)
    inner = Identity + snr_linear * (H @ H.conj().T)
    sign, logdet = np.linalg.slogdet(inner)
    return logdet / np.log(2)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"--- Calculating Multi-RT Achievable Rate ({CENTER_FREQ_STR} GHz) ---")
    
    # 1. Load Coupling and Eigen Data
    print("Computing Coupling Matrices...")
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_R = calculate_coupling_matrix(Z_FILE_RX, Z0)
    C_T_sqrt = sqrtm(C_T)
    C_R_sqrt = sqrtm(C_R)
    
    U_T_full = load_eigenvectors(EIGEN_FILE_TX)
    U_R_full = load_eigenvectors(EIGEN_FILE_RX)
    
    # Receiver truncates to 4 (keeps all in 2x2 case)
    U_R_trunc = U_R_full[:, :R_R]

    print(f"Loading channel samples from: \n  {os.path.basename(CHANNEL_FILE)}")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_complex, (2, 1, 0)) # Reshape to (3000, 4, 49)
    num_samples = H_samples.shape[0]
    print(f"Loaded {num_samples} samples.")
    
    # 2. Initialize Result Dictionary
    rate_coupling_avg = np.zeros(len(SNR_dB_Range))
    rates_modal = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    
    # 3. Compute System Channel for all samples
    print("Computing Coupled System Channels...")
    H_c_all = np.matmul(np.matmul(C_R_sqrt, H_samples), C_T_sqrt) # (3000, 4, 49)
    
    # Pre-compute Modal Channels for each RT
    H_tilde_all = {}
    for rt in RT_LIST:
        # Note: Your new GEVD script sorted abs(lambda) in ASCENDING order. 
        # The smallest eigenvalues (lowest X/R) are at the FRONT.
        U_T_trunc = U_T_full[:, :rt] 
        
        term1 = np.matmul(U_R_trunc.conj().T[None, :, :], H_c_all) 
        H_tilde_all[rt] = np.matmul(term1, U_T_trunc[None, :, :]) 
    
    # 4. Processing Loop (SNR Sweep)
    print("Sweeping SNR...")
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        c_full_list = []
        c_modal_lists = {rt: [] for rt in RT_LIST}
        
        for k in range(num_samples):
            # Full Channel Capacity
            c_full_list.append(capacity_logdet(H_c_all[k], snr_linear))
            
            # Modal Channel Capacities
            for rt in RT_LIST:
                c_modal_lists[rt].append(capacity_logdet(H_tilde_all[rt][k], snr_linear))
            
        rate_coupling_avg[i] = np.mean(c_full_list)
        for rt in RT_LIST:
            rates_modal[rt][i] = np.mean(c_modal_lists[rt])
            
        # Dynamic print formatting based on RT_LIST size
        rt_str = " | ".join([f"R_T={rt}: {rates_modal[rt][i]:.4f}" for rt in RT_LIST])
        print(f"SNR {snr_db:>3} dB | Full: {rate_coupling_avg[i]:.4f} | {rt_str}")

    # 5. Plotting
    plt.figure(figsize=(10, 7))
    # plt.plot(SNR_dB_Range, rate_coupling_avg, 'b-o', label=f'Full Coupling Aware ($N_T={N_T}$)', linewidth=2.5)
    
    # Plotting different RT lines with distinct colors and markers
    colors = ['orange', 'green', 'red', 'purple', 'cyan']
    markers = ['^', 's', 'D', 'v', 'p']
    
    for idx, rt in enumerate(RT_LIST):
        plt.plot(SNR_dB_Range, rates_modal[rt], 
                 color=colors[idx % len(colors)], marker=markers[idx % len(markers)], linestyle='--', 
                 label=f'Modal Domain ($r_T={rt}$ smallest EVs)', linewidth=2)
    
    plt.title(f'Achievable Rate vs SNR ({CENTER_FREQ_STR} GHz)\nTx: {TX_DIM}, Rx: {RX_DIM}', fontsize=14)
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Rate [bps/Hz]', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save Filename
    filename = f"Rate_{CENTER_FREQ_STR}GHz_Multi_RT_comparison.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to: {save_path}")

if __name__ == "__main__":
    main()