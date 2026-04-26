import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
# Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to your newly generated PyTorch tensor dataset
DATASET_PATH = os.path.join(SCRIPT_DIR, "training_testing_dataset", "x0_12GHz_8x8Tx_1x1Rx_1000000samples_rT38.pt")

def plot_distribution_verification(complex_data):
    """
    Plots the empirical histogram of the Real and Imaginary data against theoretical Gaussian curves.
    """
    print("Calculating statistics and generating plots...")
    
    # Extract components
    real_part = np.real(complex_data)
    imag_part = np.imag(complex_data)

    # Create a wide figure for two plots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------------------------------------------------------
    # Plot 1: The Real Part vs. Theoretical Gaussian
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.hist(real_part, bins=150, density=True, alpha=0.6, color='orange', label='Data distribution (Real)')
    
    # Fit and draw the Gaussian curve
    mu_real, std_real = stats.norm.fit(real_part)
    xmin_r, xmax_r = ax1.get_xlim()
    x_axis_r = np.linspace(xmin_r, xmax_r, 200)
    ax1.plot(x_axis_r, stats.norm.pdf(x_axis_r, mu_real, std_real), 'r--', linewidth=2.5, 
             label=f'Gaussian Fit\n($\mu$={mu_real:.4f}, $\sigma$={std_real:.4f})')
    
    ax1.set_title(f'Real Part Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Plot 2: The Imaginary Part vs. Theoretical Gaussian
    # ---------------------------------------------------------
    ax2 = axes[1]
    ax2.hist(imag_part, bins=150, density=True, alpha=0.6, color='dodgerblue', label='Data distribution (Imag)')
    
    # Fit and draw the Gaussian curve
    mu_imag, std_imag = stats.norm.fit(imag_part)
    xmin_i, xmax_i = ax2.get_xlim()
    x_axis_i = np.linspace(xmin_i, xmax_i, 200)
    ax2.plot(x_axis_i, stats.norm.pdf(x_axis_i, mu_imag, std_imag), 'r--', linewidth=2.5, 
             label=f'Gaussian Fit\n($\mu$={mu_imag:.4f}, $\sigma$={std_imag:.4f})')

    ax2.set_title(f'Imaginary Part Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Save the output to the exact same directory as the script
    plt.tight_layout()
    output_filename = os.path.join(SCRIPT_DIR, 'distribution_verification.png')
    plt.savefig(output_filename, dpi=300)
    print(f"\nSuccess! Plot saved directly to: {output_filename}")

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find dataset at {DATASET_PATH}")
        return
        
    print(f"Loading dataset from {DATASET_PATH}...")
    x0_tensor = torch.load(DATASET_PATH)
    
    # Extract Mode 0 (the strongest mode) for visualization
    # mode_idx = 10
    # x0_mode = x0_tensor[:, mode_idx].numpy()
    x0_mode = x0_tensor.mean(dim=1).numpy()
    
    plot_distribution_verification(x0_mode)

if __name__ == "__main__":
    main()