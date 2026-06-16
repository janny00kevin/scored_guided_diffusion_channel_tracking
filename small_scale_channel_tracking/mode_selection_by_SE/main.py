import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import sqrtm
from functions import (calculate_coupling_matrix, generate_modal_precoder, 
                       generate_port_precoder, calculate_user_rate)

# Suppress potential multi-threading conflicts in libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. File Paths and Basic Configuration ---
Z_FILE = "8x8_UPA_12GHz_Z.mat"
EIGEN_FILE = "8x8_UPA_12GHz_eigen.mat"
H_FILE = "H_UMa_rojer.mat"  # Using real channel file with perfect CSI

# --- 2. Simulation Parameters ---
RT_LIST = [64, 40, 28]  
SNR_DB_RANGE = np.arange(-10, 15, 2)  # SNR range from -10 dB to 14 dB
MC_count = 1
P_tx_dbm = 46
P_linear = 10**((P_tx_dbm - 30) / 10) 

# --- 3. Pre-calculate Hardware Parameters ---
# Compute coupling matrix and its square root for precoding
Ct = calculate_coupling_matrix(Z_FILE)
Ct_sqrt, _ = la.sqrtm(Ct, disp=False)
Ct_inv_sqrt = np.linalg.pinv(Ct_sqrt)

# Load eigen-basis for modal precoding
data_e = sio.loadmat(EIGEN_FILE)
Ut_full = data_e['U_T_sorted']

# Load and normalize real channel samples
H_raw = sio.loadmat(H_FILE)['H'] 
avg_power = np.mean(np.abs(H_raw)**2)
H_samples = H_raw / np.sqrt(avg_power) 

# Initialize result storage
results_modal_dict = {rt: [] for rt in RT_LIST}
result_port = []

# --- 4. Simulation Core Loop ---
for snr_db in SNR_DB_RANGE:
    # Initialize accumulators for each SNR iteration
    temp_sum_port = 0
    temp_sum_modal = {rt: 0 for rt in RT_LIST}
    
    sigma2 = 1 / (10**(snr_db / 10))
    
    for i in range(MC_count):
        # Apply coupling to obtain effective channel Hc
        Hc = H_samples[i, :, :] @ Ct_sqrt
        
        # A. Compute Port-domain baseline (Full-Digital SVD on Hc)
        W_p = generate_port_precoder(Hc)
        W_p = W_p / np.linalg.norm(W_p, 'fro') * np.sqrt(P_linear)  # Enforce transmit power constraint
        temp_sum_port += calculate_user_rate(Hc, W_p, W_p, sigma2)
        
        # B. Compute Modal-domain rates for different rT values
        for rt in RT_LIST:
            # Truncate basis to the top 'rt' eigen-modes
            Ut_trunc = Ut_full[:, :rt] 
            
            # Modal precoding with power normalization
            W_m = generate_modal_precoder(Hc, Ct, Ut_trunc)
            W_m = W_m / np.linalg.norm(W_m, 'fro') * np.sqrt(P_linear)  
            temp_sum_modal[rt] += calculate_user_rate(Hc, W_m, W_m, sigma2)
            
    # Store averaged results
    result_port.append(temp_sum_port / MC_count)
    for rt in RT_LIST:
        results_modal_dict[rt].append(temp_sum_modal[rt] / MC_count)
        
    print(f"SNR {snr_db} dB simulation completed...")

# --- 5. Plotting: Rate Comparison Curves ---
plt.figure(figsize=(10, 6))
plt.plot(SNR_DB_RANGE, result_port, 'k-o', linewidth=2.5, label='Port-domain Full-Digital Bound')

markers = ['s', '^', 'v', 'd', 'p', '*'] 
for idx, rt in enumerate(RT_LIST):
    m = markers[idx % len(markers)]
    plt.plot(SNR_DB_RANGE, results_modal_dict[rt], f'--{m}', label=rf'Modal-domain ($r_T$={rt})')

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Average Rate (bps/Hz)', fontsize=12)
plt.title('Average Sum Rate Comparison (Perfect CSI)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')
plt.savefig('rate_comparison_curves.png', bbox_inches='tight', dpi=300)

# --- 6. Plotting: Modal Efficiency Percentage Table ---
table_data = []
for snr_idx in range(len(SNR_DB_RANGE)):
    row_percent = []
    port_val = result_port[snr_idx]
    for rt in RT_LIST:
        modal_val = results_modal_dict[rt][snr_idx]
        percent = (modal_val / port_val) * 100
        row_percent.append(f"{percent:.2f}%") 
    table_data.append(row_percent)

fig_table, ax_table = plt.subplots(figsize=(10, 6))
ax_table.axis('tight')
ax_table.axis('off') 

col_labels = [f'Modal (rT={rt})' for rt in RT_LIST]
row_labels = [f'{snr} dB' for snr in SNR_DB_RANGE]

the_table = ax_table.table(cellText=table_data,
                           rowLabels=row_labels,
                           colLabels=col_labels,
                           rowColours=["#f1f1f1"] * len(row_labels),
                           colColours=["#cfe2f3"] * len(col_labels), 
                           cellLoc='center',
                           loc='center')

the_table.auto_set_font_size(False)
the_table.set_fontsize(11)    
the_table.scale(1.2, 1.6)      

plt.title('Modal Efficiency Relative to Port-domain Bound (%)', y=1.05, fontweight='bold', fontsize=13)
plt.savefig('modal_efficiency_table.png', bbox_inches='tight', dpi=300)
plt.show()