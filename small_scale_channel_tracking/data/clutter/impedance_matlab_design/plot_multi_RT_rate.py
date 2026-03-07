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

MODES = [2.065, 38.75]
RT_LIST = [1, 2, 3, 49]
SNR_dB_Range = np.arange(-4, 12, 2)
Z0 = 50 

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(z_mat_path, z0=50):
    data = sio.loadmat(z_mat_path)
    Z = data['Z_matrix']
    N = Z.shape[0]
    Term_A = Z + z0 * np.eye(N)
    X = np.linalg.solve(Term_A, Z)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N))
    return 0.5 * np.real(X @ Y)

def capacity_miso(H, snr_linear):
    # Capacity for 49x1 MISO: log2(1 + SNR * ||h||^2)
    h_norm_sq = np.sum(np.abs(H)**2)
    return np.log2(1 + snr_linear * h_norm_sq)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    for freq in MODES:
        print(f"\n--- Calculating MISO Achievable Rates ({freq} GHz) ---")
        
        # Link accurate physics frequency to the generated channel data label
        chan_freq_str = "39" if freq == 38.75 else "2"
        channel_file = os.path.join(CHAN_DIR, f"channel_data_{chan_freq_str}GHz_7x7Tx_1x1Rx_3000samples.mat")
        
        z_file = os.path.join(SCRIPT_DIR, "Z_results", f"7x7_UPA_{freq:.3f}GHz_Z.mat")
        eigen_file = os.path.join(SCRIPT_DIR, "eigen_result", f"7x7_UPA_{freq:.3f}GHz_eigen.mat")
        
        if not os.path.exists(channel_file):
            print(f"[Warning] MISO Channel file not found: {channel_file}")
            continue
        if not os.path.exists(z_file) or not os.path.exists(eigen_file):
            print(f"[Warning] Z-matrix or Eigen file missing for {freq} GHz. Run MATLAB scripts first.")
            continue
            
        print("Computing transmit coupling matrix...")
        C_T = calculate_coupling_matrix(z_file, Z0)
        C_T_sqrt = sqrtm(C_T)
        
        print("Loading sorted eigenvectors (Ascending)...")
        U_T_sorted = sio.loadmat(eigen_file)['U_T_sorted']
        
        with h5py.File(channel_file, 'r') as f:
            h_raw = f['H_samples'][()]
            H_complex = h_raw['real'] + 1j * h_raw['imag']
            
        # MATLAB output is (49, 1, 3000) -> Reshape to (3000, 1, 49)
        H_samples = np.transpose(H_complex, (2, 1, 0)) 
        num_samples = H_samples.shape[0]
        
        rate_full = np.zeros(len(SNR_dB_Range))
        rates_modal = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
        
        # Coupled physical channel
        H_c_all = np.matmul(H_samples, C_T_sqrt) # (3000, 1, 49)
        
        # Pre-project modal channels based on smallest rT eigenvalues
        H_tilde_all = {}
        for rt in RT_LIST:
            U_T_trunc = U_T_sorted[:, :rt] 
            H_tilde_all[rt] = np.matmul(H_c_all, U_T_trunc) # (3000, 1, rt)
            
        print("Sweeping SNR levels...")
        for i, snr_db in enumerate(SNR_dB_Range):
            snr_linear = 10**(snr_db / 10)
            
            c_f = []
            c_m = {rt: [] for rt in RT_LIST}
            
            for k in range(num_samples):
                c_f.append(capacity_miso(H_c_all[k], snr_linear))
                for rt in RT_LIST:
                    c_m[rt].append(capacity_miso(H_tilde_all[rt][k], snr_linear))
                    
            rate_full[i] = np.mean(c_f)
            for rt in RT_LIST:
                rates_modal[rt][i] = np.mean(c_m[rt])
                
            rt_str = " | ".join([f"r_T={rt}: {rates_modal[rt][i]:.4f}" for rt in RT_LIST])
            print(f"SNR {snr_db:>3} dB | Full(49): {rate_full[i]:.4f} | {rt_str}")

        # Plotting
        plt.figure(figsize=(10, 7))
        colors = ['orange', 'green', 'red', 'purple']
        markers = ['^', 's', 'D', 'v']
        
        for idx, rt in enumerate(RT_LIST):
            plt.plot(SNR_dB_Range, rates_modal[rt], color=colors[idx % len(colors)], 
                     marker=markers[idx % len(markers)], linestyle='--', 
                     label=f'Modal Domain ($r_T={rt}$ smallest EVs)', linewidth=2)
                     
        plt.title(f'MISO Achievable Rate vs SNR ({freq} GHz)\nTx: [7, 7], Rx: [1, 1]', fontsize=14)
        plt.xlabel('SNR [dB]', fontsize=12)
        plt.ylabel('Rate [bps/Hz]', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        save_path = os.path.join(OUTPUT_DIR, f"Rate_{freq:.3f}GHz_MISO_Multi_RT.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()