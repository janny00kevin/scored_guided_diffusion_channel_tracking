import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm, logm, det
import os
import h5py
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Select Mode ---
MODE = 3  # 1: 2.0625, 2: 28, 3: 38.75 GHz

if MODE == 1:
    FREQ_GHZ = 2
    R_T = 6
elif MODE == 2:
    FREQ_GHZ = 28
    R_T = 27
elif MODE == 3:
    FREQ_GHZ = 39
    R_T = 6

# Fixed Parameters
ASF = 2
TX_DIM = [7, 7]  # 49
RX_DIM = [2, 2]  # 4
N_T = TX_DIM[0] * TX_DIM[1]
N_R = RX_DIM[0] * RX_DIM[1]
R_R = 4  # Receiver Rank
Z0 = 50 

# SNR Range (dB)
SNR_dB_Range = np.arange(-4, 12, 2) # [-4, -2, 0, 2, ..., 10]

# Paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

Z_FILE_TX = os.path.join(DATA_DIR, "impedance_matrix_matlab", "Z_results", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_Z.mat")
Z_FILE_RX = os.path.join(DATA_DIR, "impedance_matrix_matlab", "Z_results", f"{RX_DIM[0]}x{RX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_Z.mat")
EIGEN_FILE_TX = os.path.join(DATA_DIR, "impedance_matrix_matlab", "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_eigen.mat")
EIGEN_FILE_RX = os.path.join(DATA_DIR, "impedance_matrix_matlab", "eigen_result", f"{RX_DIM[0]}x{RX_DIM[1]}_UPA_{FREQ_GHZ}GHz_spacing{ASF}_eigen.mat")
CHANNEL_FILE = os.path.join(DATA_DIR, "channel", f"channel_data_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_3000samples.mat")

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(z_mat_path, z0=50):
    if not os.path.exists(z_mat_path): raise FileNotFoundError(f"Not found: {z_mat_path}")
    data = sio.loadmat(z_mat_path)
    Z = data['Z_matrix']
    N = Z.shape[0]
    Term_A = Z + z0 * np.eye(N)
    # C = 0.5 * real( inv(A) * Z * inv(A^H) )
    X = np.linalg.solve(Term_A, Z)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N))
    return 0.5 * np.real(X @ Y)

def load_eigenvectors(eigen_path):
    if not os.path.exists(eigen_path): raise FileNotFoundError(f"Not found: {eigen_path}")
    return sio.loadmat(eigen_path)['U_T_sorted']

def capacity_logdet(H, snr_linear):
    # Capacity C = log2( det( I + SNR * H * H^H ) )
    # Assuming Equal Power Allocation for simplicity here
    # Check dimensions
    N_Rx, N_Tx = H.shape
    # For stability, usually H is normalized or SNR includes transmit power P
    # Here we treat 'snr_linear' as eta (P/sigma^2)
    
    Identity = np.eye(N_Rx)
    # Using np.linalg.slogdet is safer for determinants
    inner = Identity + snr_linear * (H @ H.conj().T)
    sign, logdet = np.linalg.slogdet(inner)
    
    # Convert natural log to log base 2: log2(x) = ln(x) / ln(2)
    return logdet / np.log(2)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"--- Calculating Achievable Rate ({FREQ_GHZ} GHz) ---")
    
    # 1. Load Data
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_R = calculate_coupling_matrix(Z_FILE_RX, Z0)
    C_T_sqrt = sqrtm(C_T)
    C_R_sqrt = sqrtm(C_R)
    
    U_T_full = load_eigenvectors(EIGEN_FILE_TX)
    U_R_full = load_eigenvectors(EIGEN_FILE_RX)
    
    U_T_trunc = U_T_full[:, :R_T]
    U_R_trunc = U_R_full[:, :R_R]

    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_complex, (2, 1, 0)) # (3000, 4, 49)
    num_samples = H_samples.shape[0]

    print(f"Loaded {num_samples} samples.")
    
    # 2. Initialize Result Arrays
    rate_coupling_avg = np.zeros(len(SNR_dB_Range))
    rate_modal_avg = np.zeros(len(SNR_dB_Range))
    
    # 3. Processing Loop (SNR Sweep)
    print("Sweeping SNR...")
    
    # Prepare H_sys and H_tilde for all samples to speed up
    # Note: For strict capacity avg, we avg the capacity of each sample, not the channel avg
    
    H_c_all = np.matmul(np.matmul(C_R_sqrt, H_samples), C_T_sqrt) # (3000, 4, 49)
    
    # Modal Channel: H_tilde = U_R_trunc^H * H _c* U_T_trunc
    # Broadcasting: (R_R, 4) @ (3000, 4, 49) @ (49, R_T)
    term1 = np.matmul(U_R_trunc.conj().T[None, :, :], H_c_all) # (3000, R_R, 49)
    H_tilde_all = np.matmul(term1, U_T_trunc[None, :, :]) # (3000, R_R, R_T)
    
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        c_full_list = []
        c_modal_list = []
        
        for k in range(num_samples):
            # Full Channel Capacity
            rate_f = capacity_logdet(H_c_all[k], snr_linear)
            c_full_list.append(rate_f)
            
            # Modal Channel Capacity
            # Note: For fairness, we compare the capacity of the REDUCED dimension channel
            # which represents the information flow limit of the truncated system.
            rate_m = capacity_logdet(H_tilde_all[k], snr_linear)
            c_modal_list.append(rate_m)
            
        rate_coupling_avg[i] = np.mean(c_full_list)
        rate_modal_avg[i] = np.mean(c_modal_list)
        
        print(f"SNR {snr_db:>3} dB | Full: {rate_coupling_avg[i]:.4f} | Modal: {rate_modal_avg[i]:.4f} bps/Hz")

    # 4. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(SNR_dB_Range, rate_coupling_avg, 'b-o', label=f'Full Coupling Aware ($N_T={N_T}$)', linewidth=2)
    plt.plot(SNR_dB_Range, rate_modal_avg, 'r--s', label=f'Modal Domain ($R_T={R_T}$)', linewidth=2)
    
    plt.title(f'Achievable Rate vs SNR ({FREQ_GHZ} GHz)\nTx: {TX_DIM}, Rx: {RX_DIM}')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Rate [bps/Hz]')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save Filename
    filename = f"Rate_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_RT{R_T}_RR{R_R}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()