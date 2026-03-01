import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm, eig
import os
import h5py
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAN_DIR = os.path.join(SCRIPT_DIR, "..", "channel")

FREQ = 38.75
CHAN_FREQ_STR = "39"  # Link to your existing 39GHz channel data
RT_LIST = [1, 2, 3, 49]
SNR_dB_Range = np.arange(-4, 12, 2)
Z0 = 50 
N_T = 49

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Helper Functions
# ==========================================
def capacity_miso(H, snr_linear):
    # Capacity for 49x1 MISO: log2(1 + SNR * ||h||^2)
    h_norm_sq = np.sum(np.abs(H)**2)
    return np.log2(1 + snr_linear * h_norm_sq)

def generate_synthetic_Z():
    """ Generates Z = 50*I + Noise + Distance-decaying Mutual Coupling """
    np.random.seed(42) # For reproducible noise
    
    # 1. Create 7x7 grid coordinates to calculate physical distances
    coords = np.array([[r, c] for r in range(7) for c in range(7)])
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    # 2. Model "Slight" Mutual Coupling (Block-Toeplitz equivalent)
    # Assume adjacent elements couple with 2 - 5j Ohms, decaying with distance
    Z_adj = 2.0 - 5.0j
    Z_mc = np.zeros((N_T, N_T), dtype=complex)
    mask = dist_matrix > 0
    # Decay amplitude by distance, and rotate phase by distance
    Z_mc[mask] = Z_adj * (1.0 / dist_matrix[mask]) * np.exp(-1j * np.pi * dist_matrix[mask])
    
    # 3. Model 50*I + Noise
    Z_ideal = 50 * np.eye(N_T, dtype=complex)
    noise_real = np.random.normal(0, 0.5, N_T) # 0.5 Ohm variance
    noise_imag = np.random.normal(0, 0.5, N_T)
    Z_diag = Z_ideal + np.diag(noise_real + 1j * noise_imag)
    
    # Combine
    Z_synthetic = Z_diag + Z_mc
    return Z_synthetic

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"\n--- Simulating Synthetic Z (50I + Noise + MC) for {FREQ} GHz ---")
    
    Z_synth = generate_synthetic_Z()
    R_T = np.real(Z_synth)
    X_T = np.imag(Z_synth)
    
    # Print sample to verify
    print("\nTop Left 3x3 Block of the Synthetic Z-Matrix:")
    print(np.round(Z_synth[:3, :3], 2))
    
    # 1. Perform GEVD (X * U = R * U * Lambda)
    lambda_vals, U_raw = eig(X_T, R_T)
    U_norm = U_raw / np.linalg.norm(U_raw, axis=0)
    
    # 2. Create TWO sorted versions (Smallest EVs vs Largest EVs)
    magnitudes = np.abs(lambda_vals)
    sort_idx_asc = np.argsort(magnitudes)        # Smallest first
    sort_idx_desc = np.argsort(magnitudes)[::-1] # Largest first
    
    U_smallest = U_norm[:, sort_idx_asc]
    U_largest = U_norm[:, sort_idx_desc]
    
    # 3. Compute the Coupling Matrix (C_T)
    Term_A = Z_synth + Z0 * np.eye(N_T)
    X = np.linalg.solve(Term_A, Z_synth)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N_T))
    C_T = 0.5 * np.real(X @ Y)
    C_T_sqrt = sqrtm(C_T)
    
    # 4. Load your existing MISO Spatial Channel
    channel_file = os.path.join(CHAN_DIR, f"channel_data_{CHAN_FREQ_STR}GHz_7x7Tx_1x1Rx_3000samples.mat")
    if not os.path.exists(channel_file):
        print(f"[Error] Channel file not found: {channel_file}")
        return
        
    print(f"\nLoading spatial channel: {os.path.basename(channel_file)}")
    with h5py.File(channel_file, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
    
    H_samples = np.transpose(H_complex, (2, 1, 0)) # (3000, 1, 49)
    num_samples = H_samples.shape[0]
    
    # Coupled physical channel (H * C_T^(1/2))
    H_c_all = np.matmul(H_samples, C_T_sqrt) 
    
    # 5. Evaluate Achievable Rates for BOTH sorting strategies
    rate_full = np.zeros(len(SNR_dB_Range))
    rates_smallest = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    rates_largest = {rt: np.zeros(len(SNR_dB_Range)) for rt in RT_LIST}
    
    # Pre-project modal channels
    H_tilde_small = {rt: np.matmul(H_c_all, U_smallest[:, :rt]) for rt in RT_LIST}
    H_tilde_large = {rt: np.matmul(H_c_all, U_largest[:, :rt]) for rt in RT_LIST}
        
    print("\nSweeping SNR levels...")
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        c_f = []
        c_s = {rt: [] for rt in RT_LIST}
        c_l = {rt: [] for rt in RT_LIST}
        
        for k in range(num_samples):
            c_f.append(capacity_miso(H_c_all[k], snr_linear))
            for rt in RT_LIST:
                c_s[rt].append(capacity_miso(H_tilde_small[rt][k], snr_linear))
                c_l[rt].append(capacity_miso(H_tilde_large[rt][k], snr_linear))
                
        rate_full[i] = np.mean(c_f)
        for rt in RT_LIST:
            rates_smallest[rt][i] = np.mean(c_s[rt])
            rates_largest[rt][i] = np.mean(c_l[rt])

    # 6. Plotting Function
    def plot_rates(rates_dict, title_modifier, save_name):
        plt.figure(figsize=(10, 7))
        colors = ['orange', 'green', 'red', 'purple']
        markers = ['^', 's', 'D', 'v']
        
        for idx, rt in enumerate(RT_LIST):
            plt.plot(SNR_dB_Range, rates_dict[rt], color=colors[idx % len(colors)], 
                     marker=markers[idx % len(markers)], linestyle='--', 
                     label=f'Modal Domain ($r_T={rt}$)', linewidth=2)
                     
        plt.title(f'Synthetic Z Achievable Rate vs SNR ({FREQ} GHz)\n{title_modifier}', fontsize=14)
        plt.xlabel('SNR [dB]', fontsize=12)
        plt.ylabel('Rate [bps/Hz]', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        save_path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    print("\nGenerating Plots...")
    plot_rates(rates_smallest, "Sorting: SMALLEST Eigenvalues (Best Matched)", f"Rate_Synth_Z_SMALLEST_{FREQ}GHz.png")
    plot_rates(rates_largest, "Sorting: LARGEST Eigenvalues (Worst Matched)", f"Rate_Synth_Z_LARGEST_{FREQ}GHz.png")

if __name__ == "__main__":
    main()