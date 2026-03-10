import numpy as np
from scipy.linalg import sqrtm
import scipy.io as sio
import os
import h5py
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# ==========================================
# 1. Configuration
# ==========================================
CENTER_FREQ_STR = "12.45"
CHANNEL_FREQ_GHZ = 12

# --- CHANNEL MODE SELECTION ---
# Set to 'UMa' to use your .mat files, or 'rayleigh' for i.i.d. Gaussian
CHANNEL_MODE = 'UMa'  # Defaulted to Rayleigh unless you have already generated the UMa channel data for 12.45GHz

TX_DIM = [8, 8]  # Updated to 8x8
RX_DIM = [1, 1]
N_T = TX_DIM[0] * TX_DIM[1]  # 64 antennas

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)
RT_LIST = [38, N_T]  # Subsets to test

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")  

# Point to the single unified .pt file you just created
EIGEN_FILE_PT = os.path.join(SCRIPT_DIR, "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_eigen.pt")

# Default filename for import mode (if UMa is used)
CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_1x1Rx_3000samples.mat")

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(Z, z0=50):
    """Calculates coupling matrix C_T directly from the Z-matrix array"""
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
    print(f"Channel Mode: {CHANNEL_MODE.upper()}")
    
    # 1. Load the Coupled System Physics from PyTorch .pt file
    print(f"Loading Physics Data from: {os.path.basename(EIGEN_FILE_PT)}...")
    if not os.path.exists(EIGEN_FILE_PT):
        print(f"[Error] PyTorch Eigen file not found: {EIGEN_FILE_PT}")
        return
        
    data = torch.load(EIGEN_FILE_PT, weights_only=True)
    Z_matrix = data['Z_matrix'].numpy()
    U_T_full = data['U_T_sorted'].numpy()
    
    print("Computing Coupling Matrix...")
    C_T = calculate_coupling_matrix(Z_matrix, Z0)
    C_T_sqrt = sqrtm(C_T)
    
    # 2. Channel Selection Logic
    if CHANNEL_MODE == 'UMa':
        print(f"Loading Channel Data from: {os.path.basename(CHANNEL_FILE)}...")
        if not os.path.exists(CHANNEL_FILE):
            print(f"[Warning] UMa Channel file not found: {CHANNEL_FILE}")
            print("Falling back to Rayleigh Fading mode...")
            CHANNEL_MODE_RUNTIME = 'rayleigh'
        else:
            CHANNEL_MODE_RUNTIME = 'UMa'
            with h5py.File(CHANNEL_FILE, 'r') as f:
                h_raw = f['H_samples'][()]
                H_complex = h_raw['real'] + 1j * h_raw['imag']
            H_samples = np.transpose(H_complex, (2, 1, 0)) 
    else:
        CHANNEL_MODE_RUNTIME = 'rayleigh'
        
    if CHANNEL_MODE_RUNTIME == 'rayleigh':
        print(f"Generating 3000 samples of i.i.d. Rayleigh fading channel...")
        np.random.seed(0)
        num_samples = 3000
        H_real = np.random.randn(num_samples, 1, N_T)
        H_imag = np.random.randn(num_samples, 1, N_T)
        H_samples = (H_real + 1j * H_imag) / np.sqrt(2) 
    
    # 3. Project to the effective Physical Channel
    print("Computing Effective H_c...")
    H_c_all = np.matmul(H_samples, C_T_sqrt) 
    
    # ==========================================
    # 4. RANKING MODES BY RATE (At 10 dB SNR)
    # ==========================================
    snr_eval = 10**(10 / 10)
    num_eval_samples = H_c_all.shape[0]
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
    print("Simulating Rate for Original, Smallest, and Largest RTs...")
    rates_smallest = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    rates_largest = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    rates_original = np.zeros(len(SNR_dB_Range)) # Original Rate without Modal Domain
    
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        # Original Coupled Channel (No Modal Domain)
        rates_original[i] = batch_capacity(H_c_all, snr_linear)
        
        for rt in RT_LIST:
            # Smallest eigenvalues (best matched/resonant)
            U_small = U_T_full[:, :rt]
            H_small = np.matmul(H_c_all, U_small[None, :, :])
            rates_smallest[rt][i] = batch_capacity(H_small, snr_linear)
            
            # Largest eigenvalues (worst matched/capacitive)
            U_large = U_T_full[:, -rt:]
            H_large = np.matmul(H_c_all, U_large[None, :, :])
            rates_largest[rt][i] = batch_capacity(H_large, snr_linear)

    # ==========================================
    # 6. PLOTTING
    # ==========================================
    print("Generating Plots...")
    output_dir = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
    os.makedirs(output_dir, exist_ok=True)
    
    chan_label = "Rayleigh" if CHANNEL_MODE_RUNTIME == 'rayleigh' else "UMa small scale"
    
    # --- Figure: Combined 4 Curves ---
    plt.figure(figsize=(8, 6))
    
    # 1. Original (Coupled Physical Channel)
    plt.plot(SNR_dB_Range, rates_original, color='m', marker='o', linestyle='-', label=r'$R_{\mathrm{coupling}} (H_c)$', linewidth=2)
    # 2. N_T EVs (Full Modal Domain)
    plt.plot(SNR_dB_Range, rates_smallest[N_T], color='b', marker='s', linestyle='--', label=f'All {N_T} EVs', linewidth=2)
    # 3. Smallest 
    plt.plot(SNR_dB_Range, rates_smallest[38], color='g', marker='^', linestyle='--', label='Smallest 38 EVs', linewidth=2)
    # 4. Largest 
    plt.plot(SNR_dB_Range, rates_largest[38], color='r', marker='D', linestyle='--', label='Largest 38 EVs', linewidth=2)

    plt.title(f'Rate vs SNR ({chan_label} {CENTER_FREQ_STR} GHz)\nCoupling vs Modal Domain', fontsize=14)
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Rate [bps/Hz]', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path_combined = os.path.join(output_dir, f"MISO_Rate_{CHANNEL_MODE_RUNTIME}_4Curves_{CENTER_FREQ_STR}GHz.png")
    plt.savefig(save_path_combined, dpi=300)
    plt.close()

    print(f"Plot successfully saved to:\n -> {save_path_combined}")

if __name__ == "__main__":
    main()