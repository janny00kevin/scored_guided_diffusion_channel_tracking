import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import sqrtm
from functions import (calculate_coupling_matrix, generate_modal_precoder, 
                       generate_port_precoder, generate_hybrid_limited_precoder,
                       calculate_user_rate)


# --- 2. Configuration and Paths ---
Z_FILE = "8x8_UPA_12GHz_Z.mat"
EIGEN_FILE = "8x8_UPA_12GHz_eigen.mat"
H_FILE = "H_HAT_12HZ_rojer.mat" 

RT_LIMIT_LIST = [64]  
NRF_RANGE = [1] + list(range(2, 65, 2)) 
FIXED_SNR_DB = 4 
MC_count = 1  # Number of Monte Carlo iterations
P_tx_dbm = 46
P_linear = 10**((P_tx_dbm - 30) / 10)

# --- 3. Pre-calculate Hardware Parameters ---
Ct = calculate_coupling_matrix(Z_FILE)
Ct_sqrt, _ = la.sqrtm(Ct, disp=False)
Ct_inv_sqrt = np.linalg.pinv(Ct_sqrt)

data_e = sio.loadmat(EIGEN_FILE)
Ut_full = data_e['U_T_sorted'] 
H_a = sio.loadmat(H_FILE)['H'] 
H_raw = sio.loadmat(H_FILE)['H_HAT'] 
avg_power = np.mean(np.abs(H_a)**2)
H_samples = H_raw / np.sqrt(avg_power)

# Data containers for results
res_ideal = {rt: [] for rt in RT_LIMIT_LIST}
res_hybrid_E = {rt: [] for rt in RT_LIMIT_LIST}

# Target: Container to store Port Domain full-digital baseline
res_port_baseline = []

# --- 4. Simulation Core Loop ---
sigma2 = 1 / (10**(FIXED_SNR_DB / 10))

# Step A: Compute Port Domain full-digital precoder as the ultimate baseline (N_RF = 64)
print(">>> Computing Port Domain full-digital optimization baseline...")
sum_se_port = 0
for i in range(MC_count):
    Hc = H_samples[i, :, :] @ Ct_sqrt
    # Perform SVD directly on Hc for full-digital power allocation
    W_p = generate_port_precoder(Hc, P_linear)
    sum_se_port += calculate_user_rate(Hc, W_p, W_p, sigma2)
# Baseline representing the 100% capacity reference
baseline = sum_se_port / MC_count
print(f"--- Ultimate Port Domain SE Baseline: {baseline:.4f} bps/Hz ---\n")


for rt_limit in RT_LIMIT_LIST:
    print(f">>>> Processing Modal Space r_T = {rt_limit}...")
    Ut_space = Ut_full[:, :rt_limit] 
    
    for n_rf in NRF_RANGE:
        if n_rf > rt_limit:
            res_ideal[rt_limit].append(res_ideal[rt_limit][-1])
            res_hybrid_E[rt_limit].append(res_hybrid_E[rt_limit][-1])
            continue
            
        sum_se_ideal, sum_se_E = 0, 0
        for i in range(MC_count):
            Hc = H_samples[i, :, :] @ Ct_sqrt
            Ut_trunc = Ut_space[:, :n_rf] 
            
            # (A) Ideal Modal (No Phase Shifter Constraints)
            W_i = generate_modal_precoder(Hc, Ct, Ut_trunc)
            W_i *= np.sqrt(P_linear)
            sum_se_ideal += calculate_user_rate(Hc, W_i, W_i, sigma2)
            
            # (B) Hybrid with E-matrix (Full)
            W_e = generate_hybrid_limited_precoder(Hc, Ct, Ut_trunc)
            W_e *= np.sqrt(P_linear)
            sum_se_E += calculate_user_rate(Hc, W_e, W_e, sigma2)
        
        res_ideal[rt_limit].append(sum_se_ideal / MC_count)
        res_hybrid_E[rt_limit].append(sum_se_E / MC_count)
        if n_rf % 10 == 0: print(f"    Progress: N_RF = {n_rf} completed")

# --- 5. Plotting N_RF Curve ---
plt.figure(figsize=(12, 8))
for idx, rt in enumerate(RT_LIMIT_LIST):
    plt.plot(NRF_RANGE, res_ideal[rt], '-', color='b', label=f'Modal Domain Fully Digital [$r_T$={rt}]')
    plt.plot(NRF_RANGE, res_hybrid_E[rt], '--o', color='g', label=f'Modal Domain Hybrid w/ E [$r_T$={rt}]', markersize=4)

# Plotting the black dashed line representing 100% Port Baseline
plt.axhline(y=baseline, color='k', linestyle=':', linewidth=2, label='Port Bound (Full-Digital Baseline)')

plt.xlabel('Number of RF Chains ($N_{RF}$)', fontsize=12)
plt.ylabel('Spectral Efficiency (bps/Hz)', fontsize=12)
plt.title(f'Comprehensive SE Analysis (SNR={FIXED_SNR_DB}dB, UMa)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('SE_Overall_Sweep.png', dpi=300)
plt.show()

# --- 6. Final Comparison Table (Relative to Port Domain Baseline) ---
table_data = []
for idx, nrf in enumerate(NRF_RANGE):
    row = [f"N_RF = {nrf}"]
    for rt in RT_LIMIT_LIST:
        # Retrieve values for both schemes
        val_i = res_ideal[rt][idx]
        val_e = res_hybrid_E[rt][idx]
        
        # Calculate efficiency percentage relative to Port full-digital baseline
        eff_i = (val_i / baseline) * 100
        eff_e = (val_e / baseline) * 100
        
        # Format: Absolute Value (Percentage)
        row.append(f"{val_i:.2f} ({eff_i:.1f}%)")
        row.append(f"{val_e:.2f} ({eff_e:.1f}%)")
    table_data.append(row)

# Column settings
col_labels = ['Hardware Config']
col_colors = ["#f2f2f2"]
for rt in RT_LIMIT_LIST:
    col_labels.extend([
        f'Ideal (rT={rt})', 
        f'Hybrid w/ E (rT={rt})'
    ])
    col_colors.extend(["#d9ead3", "#cfe2f3"])

fig, ax = plt.subplots(figsize=(14, 18)) 
ax.axis('off')

widths = [0.15] + [0.3] * (len(RT_LIMIT_LIST) * 2)

the_table = ax.table(cellText=table_data, 
                      colLabels=col_labels, 
                      colColours=col_colors,
                      cellLoc='center', 
                      loc='center',
                      colWidths=widths)

# --- Table Styling ---
the_table.auto_set_font_size(False)
the_table.set_fontsize(9)

for position, cell in the_table.get_celld().items():
    cell.set_height(0.022) 
    if position[0] == 0: # Header row
        cell.set_text_props(weight='bold')
        cell.set_height(0.04)

plt.title(f'Comprehensive Efficiency Analysis: {H_FILE.split("_")[1]} (Baseline: Port Domain Full-Digital Bound)', 
          y=0.97, fontweight='bold', fontsize=16)

plt.savefig('Full_Efficiency_Comparison_Table.png', bbox_inches='tight', dpi=300)
plt.show()