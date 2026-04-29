import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm
import h5py
import os
import sys
import torch

# Add the project root to the path so we can import from mode_selection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..")) 

# Import the helper function from your utils
from utils.channel_utils import calculate_coupling_matrix

# ==========================================
# 1. Configuration
# ==========================================
FREQ_GHZ = 12.45
R_T = 64
NUM_SAMPLES = 1000000  # 1000000 or 3000
TX_DIM = [8, 8]
RX_DIM = [1, 1]
MODE = 'spatial' # 'spatial' or 'modal'

# File Paths
DATA_DIR = os.path.join(SCRIPT_DIR) 
CHANNEL_FILE = os.path.join(DATA_DIR, "channel", f"channel_data_SC_{int(FREQ_GHZ)}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_SAMPLES}samples.mat")
Z_FILE_TX = os.path.join(DATA_DIR, "HFSS", "Z_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{int(FREQ_GHZ)}GHz_Z.mat")
EIGEN_FILE_TX = os.path.join(DATA_DIR, "HFSS", "eigen_result", f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{int(FREQ_GHZ)}GHz_eigen.mat")

# Output Directory & File setup
OUTPUT_DIR = os.path.join(DATA_DIR, "training_testing_dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Appending size info to the filename
if MODE == 'spatial':
    OUTPUT_FILENAME = f"x0_{int(FREQ_GHZ)}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_SAMPLES}samples_spatial.pt"
elif MODE == 'modal':
    OUTPUT_FILENAME = f"x0_{int(FREQ_GHZ)}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_SAMPLES}samples_rT{R_T}.pt"
else:
    raise ValueError(f"Invalid MODE: {MODE}. Choose 'spatial' or 'modal'.")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    print(f"--- Generating x_0 Dataset ({FREQ_GHZ} GHz, {NUM_SAMPLES} Samples) ---")
    
    # 1. Compute Transmit Coupling Matrix using your helper
    print("Computing Transmit Coupling Matrix (C_T)...")
    C_T = calculate_coupling_matrix(Z_FILE_TX)
    C_T_sqrt = sqrtm(C_T)
        
    if MODE == 'modal':
        # 2. Load Modal Eigenvectors
        print("Loading Eigenvectors...")
        U_T_full = sio.loadmat(EIGEN_FILE_TX)['U_T_sorted']
        U_T_trunc = U_T_full[:, :R_T]
    
    # 3. Load Raw Channel Samples
    print(f"Loading raw channel samples from:\n{os.path.basename(CHANNEL_FILE)}")
    with h5py.File(CHANNEL_FILE, 'r') as f:
        h_raw = f['H_samples'][()]
        H_complex = h_raw['real'] + 1j * h_raw['imag']
        
    # H_samples in MATLAB is (num_samples, 1, 64). 
    # Transposing from HDF5 format (64, 1, num_samples) to Python format (num_samples, 1, 64)
    H_samples = np.transpose(H_complex, (2, 1, 0))
    print(f"Loaded {H_samples.shape[0]} samples successfully.")
    H_c = np.matmul(H_samples, C_T_sqrt)
    
    
    # Squeeze the 1xr_T or 1xN_T matrix into a flat length-r_T vector per sample
    if MODE == 'spatial':
        # 4. Compute x_0 (Spatial Domain Projection)
        print("Using spatial domain coupling-aware channels as x_0...")
        x_0 = np.squeeze(H_c) # Shape: (1000000, N_T)
    elif MODE == 'modal':
        # 4. Compute x_0 (Modal Domain Projection)
        print("Projecting spatial channels into modal domain...")
        H_tilde = np.matmul(H_c, U_T_trunc)
        x_0 = np.squeeze(H_tilde) # Shape: (1000000, r_T)
    
# 5. Save the Dataset
    x_0_tensor = torch.tensor(x_0, dtype=torch.complex64) # <--- Convert to PyTorch tensor
    print(f"Saving dataset... Shape: {x_0_tensor.shape}, Type: {x_0_tensor.dtype}")
    
    # Save using torch instead of numpy
    torch.save(x_0_tensor, OUTPUT_FILE)
    print(f"Done! x_0 dataset saved to:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    main()