import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Configuration
# ==========================================
FREQ_GHZ = 38.75 
GRID_SIZE = [7, 7]

# Paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EIGEN_DIR = os.path.join(SCRIPT_DIR, "eigen_result")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "achievable_rate_plot")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    # Construct filename matching your workspace format
    filename = f"{GRID_SIZE[0]}x{GRID_SIZE[1]}_UPA_{FREQ_GHZ:.2f}GHz_eigen.mat"
    filepath = os.path.join(EIGEN_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[Error] Data file not found: {filepath}")
        return

    # Load the .mat file
    data = sio.loadmat(filepath)
    
    # Extract eigenvalues (sorted by magnitude in Generalized_Eig_Decomp.m)
    if 'lambda_sorted' in data:
        lambdas = data['lambda_sorted'].flatten()
    else:
        print("[Error] Variable 'lambda_sorted' not found in file.")
        return
        
    abs_lambdas = np.abs(lambdas)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Stem plot for individual mode magnitudes
    markerline, stemlines, baseline = plt.stem(
        range(1, len(abs_lambdas) + 1), 
        abs_lambdas, 
        linefmt='b-', 
        markerfmt='bo', 
        basefmt="k-"
    )
    plt.setp(markerline, 'markersize', 4)

    # Red line marked as 1
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Threshold = 1 (radiation energy vs stored energy)')

    # Title and Labels
    plt.title(f'Magnitude of Eigenvalues\n{GRID_SIZE[0]}x{GRID_SIZE[1]} UPA at {FREQ_GHZ} GHz', fontsize=14)
    plt.xlabel('Mode Index', fontsize=12)
    plt.ylabel('Absolute Value $|\lambda|$', fontsize=12)
    
    # Set Y-axis range from 0 to 10
    plt.ylim(0, 10) 
    
    # Custom ticks for clarity
    plt.yticks(np.arange(0, 11, 1))
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    
    # Save output to achievable_rate_plot/
    save_name = f"Eigenvalue_Magnitude_Linear_0to10_{FREQ_GHZ}GHz.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print("="*50)
    print(f"[Success] Plot saved to: {save_path}")
    print("="*50)

if __name__ == "__main__":
    main()