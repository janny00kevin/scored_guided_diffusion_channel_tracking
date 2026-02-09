import numpy as np
import scipy.io as sio
import os

# ==========================================
# 1. Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "impedance_matrix_matlab", "eigen_result")

# List of configurations to check
# Format: (Frequency_GHz, Rows, Cols, Spacing)
configs = [
    (2, 7, 7, 2),
    (28, 7, 7, 2),
    (39, 7, 7, 2)
]

def check_orthogonality():
    print("--- Orthogonality Check (Gram Matrix Off-Diagonal Leakage) ---")
    # Headers
    print(f"{'Freq (GHz)':<10} | {'Max Off-Diag':<15} | {'Mean Off-Diag':<15} | {'Mean Diagonal':<15} | {'Verdict'}")
    print("-" * 85)

    for freq, rows, cols, asf in configs:
        # Construct filename matching the MATLAB output format
        # Note: Filenames use %.0f for frequency, so 2.0625 -> 2
        filename = f"{rows}x{cols}_UPA_{freq}GHz_spacing{asf}_eigen.mat"
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"{str(freq):<10} | {'[FILE NOT FOUND]':<15} | {'---':<15} | {'---':<15} | N/A")
            continue

        # Load Data
        try:
            data = sio.loadmat(filepath)
            U_T = data['U_T_sorted'] 
            
            # --- Orthogonality Calculation ---
            # 1. Compute Gram Matrix: G = U^H * U
            G = U_T.conj().T @ U_T
            N = G.shape[0] # Number of modes (49)
            
            # 2. Extract Diagonal
            diag_vals = np.diag(G)
            mean_diag = np.mean(np.abs(diag_vals))
            
            # 3. Extract Off-Diagonal
            # Subtract the diagonal elements to isolate leakage
            off_diag_matrix = G - np.diag(diag_vals)
            abs_off_diag = np.abs(off_diag_matrix)
            
            # 4. Calculate Metrics
            max_leak = np.max(abs_off_diag)
            
            # Mean Off-Diagonal: Sum of magnitudes divided by number of off-diagonal elements (N^2 - N)
            mean_leak = np.sum(abs_off_diag) / (N * N - N)
            
            # Verdict
            is_orthogonal = max_leak < 1e-6
            verdict = "Unitary" if is_orthogonal else "Non-Unitary"

            print(f"{str(freq):<10} | {max_leak:<15.4f} | {mean_leak:<15.4f} | {mean_diag:<15.4f} | {verdict}")

        except Exception as e:
            print(f"{str(freq):<10} | Error: {e}")

    print("-" * 85)
    print("Definitions:")
    print(" - 'Max Off-Diag': The worst-case coupling between two distinct modes.")
    print(" - 'Mean Off-Diag': The average coupling across all pairs of distinct modes.")
    print("   (High values confirm the basis is significantly non-orthogonal).")

if __name__ == "__main__":
    check_orthogonality()