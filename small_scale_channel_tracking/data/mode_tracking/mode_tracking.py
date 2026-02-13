import numpy as np
import scipy.io as sio
import scipy.linalg as la
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Configuration
# ==========================================
# Frequencies must match your Bash script exactly
# freqs_ghz = [38.65, 38.70, 38.75, 38.80, 38.85]
freqs_ghz = [2.0125, 2.0375, 2.0625, 2.0875, 2.1125]

grid_size = [7, 7]  # 49 elements
num_modes = grid_size[0] * grid_size[1]

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Z_DIR = os.path.join(SCRIPT_DIR, "Z_results")
EIGEN_DIR = os.path.join(SCRIPT_DIR, "eigen_result")

def load_step(freq):
    """Loads U (Eigenvectors) and R (Resistance) for a specific frequency."""
    
    # 1. Load Eigenvectors (from your new MATLAB script)
    # Format: 7x7_UPA_38.65GHz_eigen.mat
    eig_name = f"{grid_size[0]}x{grid_size[1]}_UPA_{freq:.2f}GHz_eigen.mat"
    eig_path = os.path.join(EIGEN_DIR, eig_name)
    
    # 2. Load Z-Matrix (to get Resistance R)
    # Format: 7x7_UPA_38.65GHz_Z.mat
    z_name = f"{grid_size[0]}x{grid_size[1]}_UPA_{freq:.2f}GHz_Z.mat"
    z_path = os.path.join(Z_DIR, z_name)
    
    if not os.path.exists(eig_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Missing files for {freq} GHz.\nCheck: {eig_path}\nCheck: {z_path}")

    # Load Data
    eig_data = sio.loadmat(eig_path)
    z_data = sio.loadmat(z_path)
    
    U = eig_data['U_T_sorted']      # 49x49 Eigenvectors
    vals = eig_data['lambda_sorted'].flatten() # Eigenvalues (X/R)
    
    Z = z_data['Z_matrix']
    R = np.real(Z) # Resistance Matrix needed for weighting
    
    return vals, U, R

def main():
    print(f"--- Starting Signed Mode Tracking for {grid_size} Array ---")

    # Storage for tracked eigenvalues: [freq_step, mode_index]
    tracked_vals = np.zeros((len(freqs_ghz), num_modes))
    
    # 1. Initialize Reference (First Freq)
    curr_vals, curr_U, _ = load_step(freqs_ghz[0])
    if curr_vals is None:
        print("Error: Could not load first file.")
        return
    
    tracked_vals[0, :] = curr_vals
    prev_U = curr_U
    
    # --- Tracking Loop ---
    for k in range(1, len(freqs_ghz)):
        freq = freqs_ghz[k]
        
        # Load Current Step
        # Note: We need R_curr to perform the weighted correlation check
        curr_vals, curr_U, curr_R = load_step(freq)
        
        # --- IMPLEMENT EQUATION (6) ---
        # Correlation Matrix B[i, j] = | u_prev_i^H * R_curr * u_curr_j |
        
        # 1. Weight the new vectors by the resistance matrix
        weighted_curr_U = curr_R @ curr_U
        
        # 2. Compute Correlation (Inner Product)
        # Result is (N x N) matrix of correlations
        B = np.abs(prev_U.conj().T @ weighted_curr_U)
        
        # --- IMPLEMENT EQUATION (7) (Matching) ---
        new_order = np.zeros(num_modes, dtype=int)
        assigned_indices = set()
        
        # Greedy Matching: For each OLD mode, find the best NEW mode
        for i in range(num_modes):
            correlations = B[i, :] # Row i: How much Old Mode i looks like each New Mode
            
            # Sort matches by strength (highest correlation first)
            best_matches = np.argsort(correlations)[::-1]
            
            for match_idx in best_matches:
                if match_idx not in assigned_indices:
                    new_order[i] = match_idx
                    assigned_indices.add(match_idx)
                    break
        
        # Re-order the current data to match the previous history
        sorted_vals = curr_vals[new_order]
        sorted_U = curr_U[:, new_order]
        
        # Store
        tracked_vals[k, :] = sorted_vals
        prev_U = sorted_U

    # 3. Plotting ALL Modes (Signed)
    plt.figure(figsize=(14, 8))
    
    # Use Jet colormap to distinguish 49 lines
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0, 1, num_modes))
    
    for i in range(num_modes):
        # Plot each mode
        plt.plot(freqs_ghz, tracked_vals[:, i], linewidth=1.2, color=colors[i], alpha=0.8)

    # Reference Line at 0 (Resonance)
    plt.axhline(0, color='black', linewidth=2, linestyle='--', label="Resonance (X=0)")
    
    # Threshold Lines (+/- 1) suggested by Professor
    plt.axhline(1, color='gray', linewidth=1, linestyle=':')
    plt.axhline(-1, color='gray', linewidth=1, linestyle=':')

    # Formatting
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('Eigenvalue (X/R)', fontsize=14)
    plt.title(f'Characteristic Eigenvalue Evolution\n{grid_size[0]}x{grid_size[1]} UPA', fontsize=16)

    # main_ticks = [38.65, 38.70, 38.75, 38.80, 38.85]
    plt.xticks(freqs_ghz)
    
    # Use SymLog (Symmetric Log) to see detail near 0 while handling large values
    plt.yscale('symlog', linthresh=1.0) 
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Colorbar to identify indices
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_modes))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Mode Index', rotation=270, labelpad=20)

    plt.tight_layout()
    
    save_file = os.path.join(SCRIPT_DIR, f"Mode_Tracking_All_Modes_{freqs_ghz[2]:.2f}GHz.png")
    plt.savefig(save_file, dpi=300)
    print(f"Tracking complete. Plot saved to: {save_file}")
    # plt.show()

if __name__ == "__main__":
    main()