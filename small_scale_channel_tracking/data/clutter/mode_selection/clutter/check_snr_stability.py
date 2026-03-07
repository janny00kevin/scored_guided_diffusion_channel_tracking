import numpy as np
from scipy.linalg import sqrtm
import os
import h5py

#################################################
### This script is for MIMO channel, not MISO ###
#################################################

# Import from your new utils folder
from utils.channel_utils import calculate_coupling_matrix, load_eigenvectors, batch_capacity

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODE = 2  # Set to 39 for 39 GHz band, or 2 for 2 GHz band

if MODE == 39:
    CENTER_FREQ_STR = "38.75"
    CHANNEL_FREQ_GHZ = 39      
elif MODE == 2:
    CENTER_FREQ_STR = "2.06"
    CHANNEL_FREQ_GHZ = 2

TX_DIM = [7, 7]
RX_DIM = [2, 2]
N_T = TX_DIM[0] * TX_DIM[1] # 49
R_R = RX_DIM[0] * RX_DIM[1] # 4

Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# DATA_DIR = os.path.join(SCRIPT_DIR, "data", "mode_selection")
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

Z_FILE_TX = os.path.join(SCRIPT_DIR, "Z_results", f"7x7_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(SCRIPT_DIR, "eigen_result", f"7x7_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")
Z_FILE_RX = os.path.join(SCRIPT_DIR, "Z_results", f"2x2_UPA_{CENTER_FREQ_STR}GHz_Z.mat")
EIGEN_FILE_RX = os.path.join(SCRIPT_DIR, "eigen_result", f"2x2_UPA_{CENTER_FREQ_STR}GHz_eigen.mat")

CHANNEL_FILE = os.path.join(CHAN_DIR, f"channel_data_{CHANNEL_FREQ_GHZ}GHz_7x7Tx_2x2Rx_3000samples.mat")

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    print(f"--- Diagnosing SNR Stability for {CENTER_FREQ_STR} GHz ---")
    
    C_T = calculate_coupling_matrix(Z_FILE_TX, Z0)
    C_R = calculate_coupling_matrix(Z_FILE_RX, Z0)
    C_T_sqrt = sqrtm(C_T)
    C_R_sqrt = sqrtm(C_R)
    
    U_T_full = load_eigenvectors(EIGEN_FILE_TX)
    U_R_full = load_eigenvectors(EIGEN_FILE_RX)
    
    U_T_norm = U_T_full / np.linalg.norm(U_T_full, axis=0)
    U_R_norm = U_R_full[:, :R_R] / np.linalg.norm(U_R_full[:, :R_R], axis=0)

    print(f"Loading channel samples...")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    H_samples = np.transpose(H_complex, (2, 1, 0)) 
    
    print("Computing Projected Channel...")
    H_c_all = np.matmul(np.matmul(C_R_sqrt, H_samples), C_T_sqrt) 
    H_sys_Rx_projected = np.matmul(U_R_norm.conj().T[None, :, :], H_c_all)
    
    # Dictionaries to store data for later comparison
    snr_orders = {}
    snr_req_modes = {}
    
    print("Sweeping SNRs...")
    for snr_db in SNR_dB_Range:
        snr_linear = 10**(snr_db / 10)
        
        # 1. Sweep individual modes
        single_mode_rates = np.zeros(N_T)
        for m in range(N_T):
            u_m = U_T_norm[:, m:m+1] 
            H_tilde_m = np.matmul(H_sys_Rx_projected, u_m[None, :, :])
            single_mode_rates[m] = batch_capacity(H_tilde_m, snr_linear)
            
        # 2. Sort descending
        sorted_indices = np.argsort(single_mode_rates)[::-1]
        snr_orders[snr_db] = sorted_indices
        
        # 3. Find 90% threshold requirement
        H_tilde_full = np.matmul(H_sys_Rx_projected, U_T_norm[None, :, :])
        target_rate = 0.90 * batch_capacity(H_tilde_full, snr_linear)
        
        req_k = N_T
        for k in range(1, N_T + 1):
            selected_modes = sorted_indices[:k]
            U_sel = U_T_norm[:, selected_modes]
            H_tilde_sel = np.matmul(H_sys_Rx_projected, U_sel[None, :, :])
            
            if batch_capacity(H_tilde_sel, snr_linear) >= target_rate:
                req_k = k
                break
                
        snr_req_modes[snr_db] = req_k

    # ==========================================
    # 3. Print the Diagnostic Answers
    # ==========================================
    print("\n" + "="*50)
    print("                   RESULTS")
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

    # Question 2: Is the required number of modes identical?
    base_req = snr_req_modes[SNR_dB_Range[0]]
    reqs_identical = all(val == base_req for val in snr_req_modes.values())
    
    print(f"\n2. Is the required modes count for 90% identical across all SNRs? -> {reqs_identical}")
    print("   Required modes count per SNR:")
    for snr in SNR_dB_Range:
        print(f"   SNR {snr:>2} dB: Requires {snr_req_modes[snr]:>2} modes to hit 90%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()