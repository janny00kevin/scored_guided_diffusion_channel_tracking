import numpy as np
from scipy.linalg import sqrtm
import scipy.io as sio
import os
import h5py

# Import from your utils folder
from utils.channel_utils import calculate_coupling_matrix, load_eigenvectors, batch_capacity

# ==========================================
# 1. Configuration (MISO Setup)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CHANNEL MODE SELECTION ---
# Set to 'import' to use your .mat files, or 'rayleigh' for i.i.d. Gaussian
CHANNEL_MODE = 'rayleigh' 

MODE = 39  # Set to 39 or 2

if MODE == 39:
    CENTER_FREQ_STR = "38.75"
    CHANNEL_FREQ_GHZ = 39      
elif MODE == 2:
    CENTER_FREQ_STR = "2.06"
    CHANNEL_FREQ_GHZ = 2

# MISO Dimensions
TX_DIM = [7, 7]
RX_DIM = [1, 1] 
N_T = TX_DIM[0] * TX_DIM[1] # 49
NUM_SAMPLES = 3000

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# Paths
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")
Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"7x7_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"7x7_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

# Default filename for import mode
CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_7x7Tx_1x1Rx_3000samples.mat")

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    print(f"--- MISO Mode Selection Analysis ({CENTER_FREQ_STR} GHz) ---")
    print(f"Channel Mode: {CHANNEL_MODE.upper()}")

    # 1. Load Tx Coupling Data (Physics-based)
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_T_sqrt = sqrtm(C_T)
    U_T_norm = load_eigenvectors(EIGEN_FILE_TX)

    # 2. Channel Selection Logic
    if CHANNEL_MODE == 'import':
        print(f"Loading Channel Data: {os.path.basename(CHANNEL_FILE)}")
        if not os.path.exists(CHANNEL_FILE):
            print(f"[Error] Channel file not found: {CHANNEL_FILE}")
            return
        with h5py.File(CHANNEL_FILE, 'r') as f:
            h_raw = f['H_samples'][()]
            H_complex = h_raw['real'] + 1j * h_raw['imag']
        H_samples = np.transpose(H_complex, (2, 1, 0)) 
    
    else: # Rayleigh Mode
        print(f"Generating {NUM_SAMPLES} samples of i.i.d. Rayleigh fading channel...")
        np.random.seed(0)
        H_real = np.random.randn(NUM_SAMPLES, 1, N_T)
        H_imag = np.random.randn(NUM_SAMPLES, 1, N_T)
        H_samples = (H_real + 1j * H_imag) / np.sqrt(2) 
    
    # 3. Compute Coupled System Channel
    H_coupled = np.matmul(H_samples, C_T_sqrt)
    
    # 4. Analyze at 10dB SNR to get the base order
    snr_eval = 10.0
    snr_linear = 10**(snr_eval / 10)
    
    single_mode_rates = np.zeros(N_T)
    for m in range(N_T):
        u_m = U_T_norm[:, m:m+1] 
        H_tilde_m = np.matmul(H_coupled, u_m[None, :, :])
        single_mode_rates[m] = batch_capacity(H_tilde_m, snr_linear)
            
    # Sort descending by rate
    base_order = np.argsort(single_mode_rates)[::-1]
    
    # Extract corresponding eigenvalues
    raw_values = sio.loadmat(EIGEN_FILE_TX)['lambda_sorted'][base_order].flatten()

    # ==========================================
    # 3. Final Output
    # ==========================================
    print("\nSorted Mode List based on Achievable Rate:")
    for i in range(N_T):
        print(f"{i+1:>2}: Index {base_order[i]+1:>2}, Eigenvalue {raw_values[i]:.4f}")

if __name__ == "__main__":
    main()