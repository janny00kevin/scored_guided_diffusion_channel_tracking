import os
import time
import numpy as np
import scipy.io as sio
import scipy.linalg as la

# Suppress potential multi-threading conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from functions import (calculate_coupling_matrix, generate_modal_precoder, 
                       generate_port_precoder, generate_hybrid_limited_precoder,
                       calculate_user_rate)

# --- 1. Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, "SE_vs_rT_result")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# --- 2. Configuration and Paths ---
FREQ_GHZ = 38
Z_FILE = os.path.join(script_dir, "data", f"8x8_UPA_{FREQ_GHZ}GHz_Z.mat")
EIGEN_FILE = os.path.join(script_dir, "data", f"8x8_UPA_{FREQ_GHZ}GHz_eigen.mat")
H_FILE = os.path.join(script_dir, "data", f"H_HAT_{FREQ_GHZ}GHz_rojer.mat") # Imperfect CSI

RT_RANGE = [1] + list(range(2, 65, 2))  # Sweeping r_T from 1 to 64
SNR_LIST = [-4, 0, 4, 10]
MC_count = 1000
P_tx_dbm = 46 
P_linear = 10**((P_tx_dbm - 30) / 10)

# --- 3. Pre-calculate Hardware Parameters ---
Ct = calculate_coupling_matrix(Z_FILE)
Ct_sqrt, _ = la.sqrtm(Ct, disp=False)

data_e = sio.loadmat(EIGEN_FILE)
Ut_full = data_e['U_T_sorted'] 

H_a = sio.loadmat(H_FILE)['H'] 
H_raw = sio.loadmat(H_FILE)['H_HAT'] 
avg_power = np.mean(np.abs(H_a)**2)
H_samples = H_raw / np.sqrt(avg_power)

# Data containers
res_port_baseline = {}
res_ideal = {snr: [] for snr in SNR_LIST}
res_hybrid_E = {snr: [] for snr in SNR_LIST}

# --- 4. Simulation Core Loop ---
start_total = time.perf_counter()

for snr_db in SNR_LIST:
    print(f"\n======================================")
    print(f"   Simulating for SNR = {snr_db} dB")
    print(f"======================================")
    
    sigma2 = 1 / (10**(snr_db / 10))

    # Compute Port Domain full-digital baseline (Top Ceiling)
    print(">>> Computing Port Domain full-digital baseline...")
    sum_se_port = 0
    for i in range(MC_count):
        Hc = H_samples[i, :, :] @ Ct_sqrt
        W_p = generate_port_precoder(Hc)
        W_p = W_p / np.linalg.norm(W_p, 'fro') * np.sqrt(P_linear) 
        sum_se_port += calculate_user_rate(Hc, W_p, W_p, sigma2)
        
    baseline = sum_se_port / MC_count
    res_port_baseline[snr_db] = baseline
    print(f"--- Ultimate Port Domain SE Baseline: {baseline:.4f} bps/Hz ---")

    # Sweep through the Number of Modes (r_T)
    print(">>> Sweeping Modal Space (r_T)...")
    for rt in RT_RANGE:
        sum_se_ideal, sum_se_E = 0, 0
        Ut_trunc = Ut_full[:, :rt] 
        
        for i in range(MC_count):
            Hc = H_samples[i, :, :] @ Ct_sqrt
            
            # (A) Ideal Modal
            W_i = generate_modal_precoder(Hc, Ct, Ut_trunc)
            W_i = W_i / np.linalg.norm(W_i, 'fro') * np.sqrt(P_linear)
            sum_se_ideal += calculate_user_rate(Hc, W_i, W_i, sigma2)
            
            # (B) Hybrid with E-matrix
            W_e = generate_hybrid_limited_precoder(Hc, Ct, Ut_trunc)
            W_e = W_e / np.linalg.norm(W_e, 'fro') * np.sqrt(P_linear)
            sum_se_E += calculate_user_rate(Hc, W_e, W_e, sigma2)
        
        res_ideal[snr_db].append(sum_se_ideal / MC_count)
        res_hybrid_E[snr_db].append(sum_se_E / MC_count)
        
        if rt % 10 == 0 or rt == 64: 
            print(f"    Progress: r_T = {rt} completed")

end_total = time.perf_counter()
print(f"\n--- Simulation Complete! Total time: {end_total - start_total:.2f} seconds ---")

# --- 5. Save Results to .mat Format ---
save_path = os.path.join(result_dir, f"SE_vs_rT_{FREQ_GHZ}GHz_data.mat")

mat_dict = {
    'SNR_LIST': SNR_LIST,
    'RT_RANGE': RT_RANGE,
}

for snr in SNR_LIST:
    snr_str = f"n{abs(snr)}" if snr < 0 else f"p{snr}"
    mat_dict[f'baseline_snr_{snr_str}'] = res_port_baseline[snr]
    mat_dict[f'ideal_snr_{snr_str}'] = res_ideal[snr]
    mat_dict[f'hybrid_snr_{snr_str}'] = res_hybrid_E[snr]

sio.savemat(save_path, mat_dict)
print(f"Data safely saved to MATLAB format at: {save_path}")