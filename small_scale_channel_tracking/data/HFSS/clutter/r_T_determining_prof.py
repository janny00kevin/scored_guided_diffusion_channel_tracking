import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. Configuration 
# ==========================================
CENTER_FREQ_STR = "12.45"
TX_DIM = [8, 8]
N_T = TX_DIM[0] * TX_DIM[1]  # 64 antennas

def determine_rT_inverted(eigenvalues, threshold=0.90):
    """
    Determine r_T using the formula: 1 - (sum(smallest r_T) / sum(all))
    """
    # 1. Take the magnitude of the eigenvalues
    eig_mags = np.abs(eigenvalues)
    
    # 2. Sort the magnitudes in ASCENDING order (smallest first)
    sorted_eigs_asc = np.sort(eig_mags)
    
    # 3. Calculate the total sum of all eigenvalue magnitudes
    total_sum = np.sum(sorted_eigs_asc)
    
    # 4. Calculate the cumulative sum of the smallest eigenvalues
    cumulative_sum = np.cumsum(sorted_eigs_asc)
    
    # 5. Apply the requested formula
    custom_metric = 1.0 - (cumulative_sum / total_sum)
    
    # 6. Find r_T:
    # Since the metric decreases from ~1.0 down to 0.0, we want to find 
    # the maximum number of modes (largest r_T) where the metric is >= threshold.
    valid_indices = np.where(custom_metric >= threshold)[0]
    
    if len(valid_indices) > 0:
        r_T = valid_indices[-1] + 1  # +1 because index is 0-based
    else:
        # Fallback if even the first mode drops below the threshold
        r_T = 1 
        
    return r_T, custom_metric, sorted_eigs_asc

def main():
    # --- File Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EIGEN_FILE_PT = os.path.join(SCRIPT_DIR, "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{CENTER_FREQ_STR}GHz_eigen.pt")

    print(f"--- Modal Metric Analysis ({CENTER_FREQ_STR} GHz) ---")
    print(f"Loading Eigen Data from: {os.path.basename(EIGEN_FILE_PT)}...")
    
    if not os.path.exists(EIGEN_FILE_PT):
        print(f"[Error] PyTorch Eigen file not found: {EIGEN_FILE_PT}")
        return
        
    data = torch.load(EIGEN_FILE_PT, weights_only=True)
    
    # Extract the eigenvalues
    eigenvalues = data['lambda_sorted'].numpy()
    eigenvalues = eigenvalues.flatten()
    
    # Set your threshold (e.g., 0.90)
    threshold = 0.90
    
    # Calculate r_T using the new logic
    r_T, custom_metric, sorted_eigs_asc = determine_rT_inverted(eigenvalues, threshold)
    
    print("\n" + "="*50)
    print(f"Total antennas (N_T): {len(eigenvalues)}")
    print(f"Target Threshold: {threshold}")
    print(f"Determined r_T: {r_T}")
    print(f"Metric value with {r_T} modes: {custom_metric[r_T-1]:.4f}")
    print("="*50 + "\n")

    # --- Plotting the Metric ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), custom_metric, marker='o', linestyle='-', color='purple', linewidth=2)
    
    # Add a horizontal line for the threshold and a vertical line for r_T
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.axvline(x=r_T, color='g', linestyle='--', label=f'Chosen $r_T$ = {r_T}')
    
    plt.title(f'Mode Selection ({TX_DIM[0]}x{TX_DIM[1]} UPA, {CENTER_FREQ_STR} GHz)', fontsize=14)
    plt.xlabel(r'Number of Smallest $|\lambda|$ (i)', fontsize=12)
    plt.ylabel(r'$1 - \frac{\sum |\lambda_i|}{\sum |\lambda_{all}|}$', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(SCRIPT_DIR, "achievable_rate_plot")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Inverted_Eigenvalue_Metric_{CENTER_FREQ_STR}GHz.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to:\n -> {save_path}")
    
    plt.show()

if __name__ == '__main__':
    main()