import os
import numpy as np
import scipy.linalg as la
import scipy.io as sio

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Point this to your pre-extracted .mat file instead of the .m file
MAT_FILE = os.path.join(SCRIPT_DIR, "Z_result", "8x8_UPA_38GHz_Z.mat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eigen_result")

GRID_SIZE = [8, 8]  # 8x8 array

def main():
    print(f"--- Starting GEVD for {MAT_FILE} ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # ==========================================
    # 2. Load Data directly from the .mat file
    # ==========================================
    try:
        mat_data = sio.loadmat(MAT_FILE)
    except Exception as e:
        print(f"Error reading {MAT_FILE}: {e}")
        return

    # Check which variable key your matrix was saved under. 
    # Usually it is 'Z_matrix' or 'Z'. If it is just 'Z', change 'Z_matrix' below to 'Z'.
    if 'Z_matrix' in mat_data:
        Z_matrix = mat_data['Z_matrix']
    elif 'Z' in mat_data:
        Z_matrix = mat_data['Z']
    else:
        print(f"Error: Could not find impedance matrix variable ('Z_matrix' or 'Z') in {MAT_FILE}")
        print(f"Available keys inside your .mat file are: {[k for k in mat_data.keys() if not k.startswith('__')]}")
        return

    # For the frequency, we hardcode 38.75 GHz as requested for now since it's a single point file
    freq_ghz = 38.75
    print(f"Processing target frequency: {freq_ghz:.3f} GHz...")

    # Determine matrix shape dynamically
    num_ports = Z_matrix.shape[0]
    print(f" -> Matrix size detected from .mat: {num_ports}x{num_ports}")

    # Extract real and imaginary components
    R_T = np.real(Z_matrix)
    X_T = np.imag(Z_matrix)

    # ==========================================
    # 4. Generalized Eigenvalue Decomposition
    # ==========================================
    # Solve: X * U = R * U * Lambda
    lambda_vals, U_raw = la.eig(X_T, R_T)
    
    # Ensure eigenvalues are purely real 
    lambda_vals = np.real(lambda_vals)
    
    # Normalize eigenvectors (Euclidean norm = 1)
    U_norm = U_raw / np.linalg.norm(U_raw, axis=0)
    
    # Sort by magnitude of eigenvalues in Ascending order
    sort_idx = np.argsort(np.abs(lambda_vals))
    
    lambda_sorted = lambda_vals[sort_idx].reshape(-1, 1) # Column vector
    U_T_sorted = U_norm[:, sort_idx]
    
    # ==========================================
    # 5. Save Results to MATLAB .mat
    # ==========================================
    out_name = f"{GRID_SIZE[0]}x{GRID_SIZE[1]}_UPA_{int(freq_ghz)}GHz_eigen.mat"
    save_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Save dictionary directly using scipy.io
    sio.savemat(save_path, {
        'U_T_sorted': U_T_sorted,
        'lambda_sorted': lambda_sorted,
        'freqs_GHz': np.array([freq_ghz], dtype=np.float64),
        'grid_size': np.array(GRID_SIZE, dtype=np.int64),
        'Z_matrix': Z_matrix
    })
    print(f" -> Saved to: {out_name}")

    print("\n--- GEVD Script Complete ---")

if __name__ == "__main__":
    main()