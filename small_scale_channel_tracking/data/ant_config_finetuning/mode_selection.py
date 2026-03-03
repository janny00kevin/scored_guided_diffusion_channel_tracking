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

MODE = 39  # Set to 39 or 2

if MODE == 39:
    CENTER_FREQ_STR = "38.75"
    CHANNEL_FREQ_GHZ = 39      
elif MODE == 2:
    CENTER_FREQ_STR = "2.06"
    CHANNEL_FREQ_GHZ = 2

# MISO Dimensions
TX_DIM = [7, 7]
RX_DIM = [1, 1] # Changed to 1x1
N_T = TX_DIM[0] * TX_DIM[1] # 49
N_R = RX_DIM[0] * RX_DIM[1] # 1

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# Paths
# DATA_DIR = os.path.join(SCRIPT_DIR, "data", "mode_selection")
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

# Only Tx impedance/eigen data is needed for MISO
Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"7x7_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"7x7_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

# Channel filename for 1x1Rx
CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_SC_{CHANNEL_FREQ_GHZ}GHz_7x7Tx_1x1Rx_3000samples.mat")

# Output directory for rate-sorted eigen components
SORTED_EIGEN_DIR = os.path.join(SCRIPT_DIR, "eigen_result_rate_sorted")
if not os.path.exists(SORTED_EIGEN_DIR):
    os.makedirs(SORTED_EIGEN_DIR)

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    print(f"--- Diagnosing SNR Stability (MISO 49x1) for {CENTER_FREQ_STR} GHz ---")
    
    # 1. Load Tx Data Only
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_T_sqrt = sqrtm(C_T)
    
    U_T_norm = load_eigenvectors(EIGEN_FILE_TX)

    # 2. Load MISO Channel Data
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    
    # Reshape: (N_T, N_R, Samples) -> (Samples, N_R, N_T)
    # Resulting shape should be (3000, 1, 49)
    H_samples = np.transpose(H_complex, (2, 1, 0)) 

    # 3. Compute Coupled System Channel (Tx side only)
    # H_c = H_raw * C_T_sqrt
    # Shapes: (3000, 1, 49) * (49, 49) -> (3000, 1, 49)
    H_coupled = np.matmul(H_samples, C_T_sqrt)
    
    # Dictionaries to store data for later comparison
    snr_orders = {}
    snr_req_modes = {}
    
    print("Sweeping SNRs...")
    for snr_db in SNR_dB_Range:
        snr_linear = 10**(snr_db / 10)
        
        # A. Sweep individual Tx modes
        single_mode_rates = np.zeros(N_T)
        for m in range(N_T):
            # Select one Tx mode vector (49x1)
            u_m = U_T_norm[:, m:m+1] 
            
            # Effective channel for this mode: H_sys * u_m
            # Shapes: (3000, 1, 49) * (1, 49, 1) -> (3000, 1, 1)
            H_tilde_m = np.matmul(H_coupled, u_m[None, :, :])
            
            # Calculate capacity for this scalar channel
            single_mode_rates[m] = batch_capacity(H_tilde_m, snr_linear)
            
        # B. Sort descending by rate
        sorted_indices = np.argsort(single_mode_rates)[::-1]
        snr_orders[snr_db] = sorted_indices
        
        # C. Find 90% threshold requirement
        # Reference: Capacity using all 49 Tx modes
        H_tilde_full = np.matmul(H_coupled, U_T_norm[None, :, :])
        target_rate = 0.90 * batch_capacity(H_tilde_full, snr_linear)
        
        req_k = N_T
        # Cumulative sum approach (greedy selection based on sorted order)
        for k in range(1, N_T + 1):
            selected_modes = sorted_indices[:k]
            U_sel = U_T_norm[:, selected_modes]
            
            # Channel using best k modes: (3000, 1, k)
            H_tilde_sel = np.matmul(H_coupled, U_sel[None, :, :])
            
            if batch_capacity(H_tilde_sel, snr_linear) >= target_rate:
                req_k = k
                break
                
        snr_req_modes[snr_db] = req_k
        # print(f"  SNR {snr_db:>2} dB: Req {req_k} modes. Top 5: {sorted_indices[:5].tolist()}")

    # ==========================================
    # 3. Print the Diagnostic Answers
    # ==========================================
    print("\n" + "="*50)
    print("                   RESULTS (MISO)")
    print("="*50)
    
    # Question 1: Are the sorted orders identical?
    base_order = snr_orders[SNR_dB_Range[0]]
    orders_identical = True
    for snr in SNR_dB_Range[1:]:
        if not np.array_equal(base_order, snr_orders[snr]):
            orders_identical = False
            break

    print(f"\n1. Are the sorted orders identical across all SNRs? -> {orders_identical}")
    if not orders_identical:
        print("   (Showing the top 10 modes per SNR to observe differences):")
        for snr in SNR_dB_Range:
            print(f"   SNR {snr:>2} dB: {snr_orders[snr][:10].tolist()}")
    else:
        print(f"   Universal Top 10 Order: {base_order[:10].tolist()}")
        # Use .flatten() to turn [[1.2], [3.4]] into [1.2, 3.4]
    raw_values = sio.loadmat(EIGEN_FILE_TX)['lambda_sorted'][base_order].flatten()
    # Now 'val' will be a single float, and this will work:
    formatted_list = [f"{val:.4f}" for val in raw_values]
    # print(f"Corresponding eigenvalues (Top 10): {formatted_list}")
    for i in range(N_T):
        print(f"{i+1:>2}: Index {base_order[i]+1:>2}, Eigenvalue {raw_values[i]:.4f}")
    # Question 2: Is the required number of modes identical?
    base_req = snr_req_modes[SNR_dB_Range[0]]
    reqs_identical = all(val == base_req for val in snr_req_modes.values())
    
    print(f"\n2. Is the required modes count for 90% identical across all SNRs? -> {reqs_identical}")
    print("   Required modes count per SNR:")
    for snr in SNR_dB_Range:
        print(f"   SNR {snr:>2} dB: Requires {snr_req_modes[snr]:>2} modes to hit 90%")
    print("="*50 + "\n")

    # # ==========================================
    # # 4. Save the Rate-Sorted Eigen-components
    # # ==========================================
    
    # # We use the order from the highest SNR (10 dB) as our universal sorting standard
    # universal_rate_order = snr_orders[10] 

    # # Load the original `.mat` file so we preserve metadata (freqs, grid_size, etc.)
    # mat_data = sio.loadmat(EIGEN_FILE_TX)

    # # 1. Reorder the Eigenvectors (columns)
    # mat_data['U_T_sorted'] = mat_data['U_T_sorted'][:, universal_rate_order]
    
    # # 2. Flatten to a strict 1D array, sort, then reshape back to a (49, 1) column vector
    # lambda_flat = mat_data['lambda_sorted'].flatten()
    # mat_data['lambda_sorted'] = lambda_flat[universal_rate_order].reshape(-1, 1)

    # # Save to the new directory using original filename convention
    # out_filename = f"7x7_UPA_{CENTER_FREQ_STR}GHz_eigen.mat"
    # out_path = os.path.join(SORTED_EIGEN_DIR, out_filename)
    # sio.savemat(out_path, mat_data)
    
    print(f"Successfully saved re-sorted components")

if __name__ == "__main__":
    main()