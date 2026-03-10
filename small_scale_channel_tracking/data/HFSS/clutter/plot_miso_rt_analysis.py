import numpy as np
from scipy.linalg import sqrtm
import os
import matplotlib.pyplot as plt
import torch

# ==========================================
# 1. Configuration
# ==========================================
CENTER_FREQ_STR = "12.45"
CHANNEL_FREQ_GHZ = 12
CHANNEL_MODE = 'UMa'  

TX_DIM = [8, 8]
N_T = TX_DIM[0] * TX_DIM[1]  # 64 antennas
Z0 = 50 
SNR_dB_Range = np.arange(-4, 12, 2)

# --- File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Point to the .pt file you generated in the previous step
EIGEN_FILE_PT = os.path.join(SCRIPT_DIR, "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_eigen.pt")

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_coupling_matrix(Z, z0=50):
    N = Z.shape[0]
    Term_A = Z + z0 * np.eye(N)
    X = np.linalg.solve(Term_A, Z)
    Y = np.linalg.solve(Term_A.conj().T, np.eye(N))
    return 0.5 * np.real(X @ Y)

def batch_capacity(H_batch, snr_linear):
    """
    Computes capacity for a MISO channel.
    H_batch shape: (Batch, N_Rx, r_T) -> (Batch, 1, r_T)
    """
    # For a MISO channel, H * H^H is the squared magnitude sum of the vector
    mag_sq = np.sum(np.abs(H_batch)**2, axis=(1, 2)) 
    rate = np.log2(1 + snr_linear * mag_sq)
    return np.mean(rate)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print(f"--- MISO Modal Saturation Analysis ({CENTER_FREQ_STR} GHz) ---")
    
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
    
    # 2. Generate 3000 samples of i.i.d. Rayleigh fading channel
    print(f"Generating 3000 samples of i.i.d. Rayleigh fading channel...")
    np.random.seed(0)
    num_samples = 3000
    H_real = np.random.randn(num_samples, 1, N_T)
    H_imag = np.random.randn(num_samples, 1, N_T)
    H_samples = (H_real + 1j * H_imag) / np.sqrt(2) 
    
    # 3. Project to the effective Physical Channel
    print("Computing Effective Physical Channel (H_c)...")
    H_c_all = np.matmul(H_samples, C_T_sqrt) 
    
    # ==========================================
    # 4. SWEEP NUMBER OF MODES (r_T)
    # ==========================================
    print("Sweeping r_T from 1 to 64 for all SNRs...")
    
    # Matrix to hold the rate for every SNR and every r_T amount
    rates_vs_rt = np.zeros((len(SNR_dB_Range), N_T))
    
    for i, snr_db in enumerate(SNR_dB_Range):
        snr_linear = 10**(snr_db / 10)
        
        # Calculate rate incrementally adding the smallest Eigenvectors
        for rt in range(1, N_T + 1):
            U_small = U_T_full[:, :rt]
            H_small = np.matmul(H_c_all, U_small[None, :, :])
            rates_vs_rt[i, rt-1] = batch_capacity(H_small, snr_linear)

    # ==========================================
    # 5. FIND MINIMUM r_T THRESHOLDS
    # ==========================================
    full_rates = rates_vs_rt[:, -1] # The rate when using all 64 modes
    
    rt_95_percent = np.zeros(len(SNR_dB_Range), dtype=int)
    rt_99_percent = np.zeros(len(SNR_dB_Range), dtype=int)
    
    for i in range(len(SNR_dB_Range)):
        # Find the first index where the rate crosses 95% and 99% of the full rate
        idx_95 = np.argmax(rates_vs_rt[i, :] >= 0.95 * full_rates[i])
        idx_99 = np.argmax(rates_vs_rt[i, :] >= 0.99 * full_rates[i])
        
        rt_95_percent[i] = idx_95 + 1 # +1 because index 0 means 1 mode
        rt_99_percent[i] = idx_99 + 1

    print("\n--- Minimum r_T required to reach Capacity ---")
    for i, snr in enumerate(SNR_dB_Range):
        print(f"SNR = {snr:>2} dB | Full Rate: {full_rates[i]:.2f} bps/Hz | 95% requires {rt_95_percent[i]:>2} modes | 99% requires {rt_99_percent[i]:>2} modes")

    # ==========================================
    # 6. PLOTTING
    # ==========================================
    print("\nGenerating Plots...")
    output_dir = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Rate vs Number of Modes (r_T) for specific SNRs ---
    snr_indices_to_plot = [0, len(SNR_dB_Range)//2, -1] # Plot lowest, middle, and highest SNR
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for c_idx, snr_idx in enumerate(snr_indices_to_plot):
        snr = SNR_dB_Range[snr_idx]
        ax1.plot(range(1, N_T + 1), rates_vs_rt[snr_idx, :], color=colors[c_idx], linewidth=2.5, label=f'SNR = {snr} dB')
        # Mark the full capacity as a dashed horizontal line
        ax1.axhline(y=full_rates[snr_idx], color=colors[c_idx], linestyle='--', alpha=0.5)

    ax1.set_title('Rate vs Number of Modes Used ($r_T$)', fontsize=14)
    ax1.set_xlabel('Number of Smallest EVs Used ($r_T$)', fontsize=12)
    ax1.set_ylabel('Achievable Rate [bps/Hz]', fontsize=12)
    ax1.set_xlim([1, 64])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # --- Plot 2: Required r_T to reach 95% and 99% Capacity vs SNR ---
    ax2.plot(SNR_dB_Range, rt_95_percent, marker='s', color='blue', linewidth=2.5, markersize=8, label='95% of Full Rate')
    ax2.plot(SNR_dB_Range, rt_99_percent, marker='^', color='red', linewidth=2.5, markersize=8, label='99% of Full Rate')
    
    ax2.set_title('Required Modes ($r_T$) to Reach Full Capacity', fontsize=14)
    ax2.set_xlabel('SNR [dB]', fontsize=12)
    ax2.set_ylabel('Minimum Required Modes ($r_T$)', fontsize=12)
    ax2.set_ylim([1, 65])
    ax2.set_yticks(np.arange(0, 65, 8))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"rT_Saturation_Analysis_{CENTER_FREQ_STR}GHz.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot successfully saved to:\n -> {save_path}")

if __name__ == "__main__":
    main()