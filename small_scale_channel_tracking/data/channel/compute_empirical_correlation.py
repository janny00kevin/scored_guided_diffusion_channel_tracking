import os
import numpy as np
import h5py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute Empirical Spatial Correlation Matrix")
    parser.add_argument('--filename', type=str, 
                        default="channel_data_SC_12GHz_8x8Tx_1x1Rx_1000000samples.mat",
                        help="Name of the channel .mat file in the current directory")
    args = parser.parse_args()

    # Resolve the path to the current directory (channel/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, args.filename)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please ensure the script is in the 'channel/' directory alongside the .mat file.")
        return

    print(f"Loading channel data from {args.filename}...")
    try:
        with h5py.File(file_path, 'r') as f:
            h_raw = f['H_samples'][()]
            # Reconstruct complex numbers
            H_complex = h_raw['real'] + 1j * h_raw['imag']
            
        # h5py loads MATLAB data in reverse dimension order.
        # Original MATLAB shape: (Rx, Tx, Samples) -> h5py shape: (Tx, Rx, Samples)
        # Transpose to get -> (Samples, Rx, Tx)
        H_samples = np.transpose(H_complex, (2, 1, 0)) 
        
        # Flatten the spatial dimensions to shape (Samples, N_antennas)
        N_samples = H_samples.shape[0]
        H_flat = H_samples.reshape(N_samples, -1)
        
    except Exception as e:
        print(f"Failed to read HDF5 file: {e}")
        return
        
    N_ant = H_flat.shape[1]
    print(f"Data loaded successfully! Shape: {N_samples} samples, {N_ant} antennas.")
    print(f"Computing {N_ant}x{N_ant} empirical spatial correlation matrix...")
    
    # Compute R = E[h * h^H]. 
    # Since H_flat is (N_samples, N_ant), H_flat.conj().T @ H_flat computes the sum of h * h^H
    R = np.dot(H_flat.conj().T, H_flat) / N_samples
    
    # Extract Diagonal (Power per antenna)
    diag_elements = np.diag(R)
    diag_real = np.real(diag_elements) # Diagonals are strictly real in a correlation matrix
    
    # Extract Off-Diagonal (Spatial correlation between different antennas)
    mask = ~np.eye(N_ant, dtype=bool)
    off_diag_elements = R[mask]
    off_diag_abs = np.abs(off_diag_elements)
    
    # Print the Results
    print("\n" + "="*50)
    print(" EMPIRICAL CORRELATION MATRIX STATS")
    print("="*50)
    
    print("\n[Diagonal Terms] (Expected to be ~1.0 if normalized power)")
    print(f"  Mean : {np.mean(diag_real):.6f}")
    print(f"  Min  : {np.min(diag_real):.6f}")
    print(f"  Max  : {np.max(diag_real):.6f}")
    
    print("\n[Off-Diagonal Terms] (Cross-correlation magnitudes)")
    print(f"  Mean : {np.mean(off_diag_abs):.6f}")
    print(f"  Min  : {np.min(off_diag_abs):.6f}")
    print(f"  Max  : {np.max(off_diag_abs):.6f}")
    
    print("\n[Top-Left 4x4 Block of the Matrix] (Magnitudes)")
    print(np.round(np.abs(R[:4, :4]), 4))
    print("="*50)

if __name__ == "__main__":
    main()