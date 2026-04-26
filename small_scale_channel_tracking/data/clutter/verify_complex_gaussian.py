import os
import torch
import numpy as np
import scipy.stats as stats

# ==========================================
# Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this points to your newly generated PyTorch tensor dataset
DATASET_PATH = os.path.join(SCRIPT_DIR, "training_testing_dataset", "x0_12GHz_8x8Tx_1x1Rx_1000000samples_rT38.pt")

def check_complex_gaussian(complex_data):
    """
    Evaluates if a 1D array of complex numbers follows a CSCG distribution.
    """
    X = np.real(complex_data)
    Y = np.imag(complex_data)
    mag = np.abs(complex_data)
    phase = np.angle(complex_data)

    print("\n--- 1. Mean and Variance Check ---")
    print(f"Mean of Real: {np.mean(X):.6f} (Ideal: ~0)")
    print(f"Mean of Imag: {np.mean(Y):.6f} (Ideal: ~0)")
    print(f"Var of Real:  {np.var(X):.6f}")
    print(f"Var of Imag:  {np.var(Y):.6f} (Ideal: matches Var of Real)")
    print(f"Covariance:   {np.cov(X, Y)[0, 1]:.6f} (Ideal: ~0)")

    print("\n--- 2. Gaussian Test on Real/Imag (K-S Test) ---")
    # Normalize the data for the standard normal test
    ks_real = stats.kstest((X - np.mean(X)) / np.std(X), 'norm')
    ks_imag = stats.kstest((Y - np.mean(Y)) / np.std(Y), 'norm')
    print(f"Real part K-S p-value: {ks_real.pvalue:.4e} (> 0.05 implies Gaussian)")
    print(f"Imag part K-S p-value: {ks_imag.pvalue:.4e} (> 0.05 implies Gaussian)")
    
    print("\n--- 3. Rayleigh Test on Magnitude (K-S Test) ---")
    # Calculate the Rayleigh scale parameter
    scale = np.sqrt(np.mean(mag**2) / 2)
    ks_mag = stats.kstest(mag, 'rayleigh', args=(0, scale))
    print(f"Magnitude K-S p-value: {ks_mag.pvalue:.4e} (> 0.05 implies Rayleigh)")

    print("\n--- 4. Uniform Test on Phase (K-S Test) ---")
    # Normalize phase to [0, 1] for the uniform test
    normalized_phase = (phase + np.pi) / (2 * np.pi)
    ks_phase = stats.kstest(normalized_phase, 'uniform')
    print(f"Phase K-S p-value:     {ks_phase.pvalue:.4e} (> 0.05 implies Uniform)")

def main():
    print(f"--- Complex Gaussian Statistical Test ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find {DATASET_PATH}")
        return
        
    print("Loading dataset...")
    x0_tensor = torch.load(DATASET_PATH)
    
    # Extract Mode 0 (the strongest mode)
    mode_idx = 0
    x0_mode = x0_tensor[:, mode_idx].numpy()
    
    # Randomly sample 5,000 points to avoid K-S test hypersensitivity to 1 million points
    np.random.seed(42)
    sample_size = min(5000, len(x0_mode))
    x0_subset = np.random.choice(x0_mode, size=sample_size, replace=False)
    
    print(f"\nRunning formal tests on a random subset of {sample_size} samples (Mode {mode_idx})...")
    check_complex_gaussian(x0_subset)

if __name__ == "__main__":
    main()