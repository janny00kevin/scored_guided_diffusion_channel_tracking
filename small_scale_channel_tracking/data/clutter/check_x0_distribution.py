import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
# Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "training_testing_dataset", "x0_12GHz_8x8Tx_1x1Rx_1000000samples_rT38.pt")

def main():
    print(f"--- Checking Distribution of x0 ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find {DATASET_PATH}")
        return
        
    print(f"Loading dataset...")
    x0_tensor = torch.load(DATASET_PATH) # Shape: (1000000, 38)
    
    # We extract Mode 0 (the dominant mode) for clear visualization. 
    # (If we flatten all 38 modes, the weak modes will create a massive spike at 0)
    mode_idx = 0
    x0_mode = x0_tensor[:, mode_idx].numpy()
    
    # ==========================================
    # 1. Compute magnitude, mean, and std
    # ==========================================
    mag = np.abs(x0_mode)
    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    
    print(f"Mode {mode_idx} Magnitude Stats:")
    print(f"  Mean: {mag_mean:.4f}")
    print(f"  Std:  {mag_std:.4f}")
    
    # ==========================================
    # 2 & 3. Plot 1: Magnitude vs Gaussian (As Requested)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Plot the empirical histogram of the magnitude
    plt.hist(mag, bins=200, density=True, alpha=0.6, color='royalblue', label='Empirical Magnitude $|x_0|$')
    
    # Plot the Gaussian distribution using the magnitude's mean and std
    x_axis = np.linspace(0, np.max(mag), 1000)
    gaussian_fit = stats.norm.pdf(x_axis, mag_mean, mag_std)
    plt.plot(x_axis, gaussian_fit, 'r--', linewidth=2, label=f'Gaussian Fit ($\mu$={mag_mean:.2f}, $\sigma$={mag_std:.2f})')
    
    # Bonus: The mathematically correct distribution for a Complex Gaussian's magnitude is Rayleigh!
    rayleigh_scale = np.sqrt(np.mean(mag**2) / 2)
    rayleigh_fit = stats.rayleigh.pdf(x_axis, scale=rayleigh_scale)
    plt.plot(x_axis, rayleigh_fit, 'g-', linewidth=2, label='Rayleigh Fit (Theoretical)')
    
    plt.title(f"Distribution of Magnitude $|x_0|$ (Mode {mode_idx})")
    plt.xlabel("Magnitude")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file_mag = os.path.join(SCRIPT_DIR, "x0_magnitude_distribution.png")
    plt.savefig(out_file_mag, dpi=300)
    print(f"Saved magnitude plot to: {out_file_mag}")
    
    # ==========================================
    # Bonus Plot: Real Part vs Gaussian (For True Verification)
    # ==========================================
    plt.figure(figsize=(10, 6))
    real_parts = np.real(x0_mode)
    real_mean = np.mean(real_parts)
    real_std = np.std(real_parts)
    
    plt.hist(real_parts, bins=200, density=True, alpha=0.6, color='darkorange', label='Empirical Real($x_0$)')
    
    x_axis_real = np.linspace(np.min(real_parts), np.max(real_parts), 1000)
    gaussian_fit_real = stats.norm.pdf(x_axis_real, real_mean, real_std)
    plt.plot(x_axis_real, gaussian_fit_real, 'r--', linewidth=2, label=f'Gaussian Fit ($\mu$={real_mean:.2f}, $\sigma$={real_std:.2f})')
    
    plt.title(f"Distribution of Real($x_0$) (Mode {mode_idx})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file_real = os.path.join(SCRIPT_DIR, "x0_real_distribution.png")
    plt.savefig(out_file_real, dpi=300)
    print(f"Saved real-part plot to: {out_file_real}")

if __name__ == "__main__":
    main()