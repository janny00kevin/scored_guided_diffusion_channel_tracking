import os
import re
import numpy as np
import scipy.linalg as la
import torch

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Point this to your new MATLAB export file
M_FILE = os.path.join(SCRIPT_DIR, "Z_result", "12GHz_patch_8_8.m")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eigen_result")

GRID_SIZE = [8, 8]  # 8x8 array

def main():
    print(f"--- Starting GEVD for {M_FILE} ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Read the raw text of the .m file directly
    try:
        with open(M_FILE, 'r') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading {M_FILE}: {e}")
        return

    # ==========================================
    # 2. Extract Data from MATLAB Syntax
    # ==========================================
    # Find the frequency array: f = [ ... ];
    freq_matches = re.findall(r'f\s*=\s*\[(.*?)\];', text)
    freqs = []
    for match in freq_matches:
        # Extract the actual floating point numbers from the match
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', match)
        if nums:
            freqs = [float(n) for n in nums]
             
    if not freqs:
        print("Error: Could not find the frequency array 'f = [...]' in the file.")
        return
        
    print(f"Detected {len(freqs)} frequency point(s).")

    # Find the matrix blocks: Z(1,:,:) = [ ... ];
    z_blocks = re.findall(r'Z\(\s*\d+\s*,\s*:\s*,\s*:\s*\)\s*=\s*\[(.*?)\];', text, re.DOTALL)
    
    if not z_blocks:
        print("Error: Could not find any Z matrix blocks in the file.")
        return

    # ==========================================
    # 3. Process Each Frequency Block
    # ==========================================
    for idx, z_text in enumerate(z_blocks):
        # Match the frequency to the Z block
        freq = freqs[idx] if idx < len(freqs) else freqs[-1]
        freq_ghz = freq / 1e9
        print(f"\nProcessing target frequency: {freq_ghz:.3f} GHz...")
        
        # MAGIC TRICK: Clean the string so Python natively understands it
        # 1. Remove ALL whitespaces and newlines
        clean_z = re.sub(r'\s+', '', z_text)
        # 2. Replace MATLAB's imaginary unit 'i' with Python's 'j'
        clean_z = clean_z.replace('i', 'j')
        # 3. Replace MATLAB row delimiters (;) with commas (,)
        clean_z = clean_z.replace(';', ',')
        # 4. Fix double signs just in case HFSS exported them (e.g., "+-")
        clean_z = clean_z.replace('+-', '-').replace('-+', '-')
        
        # Split by comma into a list of strings
        str_vals = clean_z.split(',')
        str_vals = [v for v in str_vals if v] # Remove empty strings
        
        # Convert directly to Python complex numbers with an error catcher
        complex_vals = []
        for v in str_vals:
            try:
                complex_vals.append(complex(v))
            except ValueError:
                print(f"\n[!] CRASHED ON STRING: '{v}'")
                raise
        
        # Determine grid shape dynamically
        num_ports = int(np.sqrt(len(complex_vals)))
        print(f" -> Matrix size detected: {num_ports}x{num_ports}")
        
        # Reshape into a 2D numpy array
        Z_matrix = np.array(complex_vals).reshape((num_ports, num_ports))
        
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
        # 5. Save Results to PyTorch .pt
        # ==========================================
        out_name = f"{GRID_SIZE[0]}x{GRID_SIZE[1]}_UPA_{freq_ghz:.2f}GHz_eigen.pt"
        save_path = os.path.join(OUTPUT_DIR, out_name)
        
        torch.save({
            'U_T_sorted': torch.from_numpy(U_T_sorted),
            'lambda_sorted': torch.from_numpy(lambda_sorted),
            'freqs_GHz': torch.tensor([freq_ghz], dtype=torch.float64),
            'grid_size': torch.tensor(GRID_SIZE, dtype=torch.int64),
            'Z_matrix': torch.from_numpy(Z_matrix)
        }, save_path)
        
        print(f" -> Saved to: {out_name}")

    print("\n--- GEVD Script Complete ---")

if __name__ == "__main__":
    main()