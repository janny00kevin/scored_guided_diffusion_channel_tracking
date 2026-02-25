import torch
import os
import numpy as np

# -----------------------------
# Configurations
# -----------------------------
RUN_ID = 1
MODE = {1: 'train', 2: 'test'}.get(RUN_ID, 'train')

# Scenario Configs
FREQ_GHZ = 39
TX_DIM = [7, 7]
RX_DIM = [1, 1]
R_T = 5
NUM_SAMPLES = 1000000
CUDA = 0

# Training settings
NUM_EPOCHS = 10000
TRAIN_BATCH_SIZE = 4096 
LR = 1e-3
MODEL_TYPE = 'mlp'
VAL_SPLIT = 0.1
PATIENCE = 15

# Diffusion Process Settings
BETA_MIN = 1e-4
BETA_MAX = 0.02
T_DIFFUSION = 50.0

MODEL_WEIGHT_FILE_NAME = f"Tracker_DDIM_{FREQ_GHZ}GHz_rT{R_T}_{MODEL_TYPE}_lr{LR:.0e}.pth"

# -----------------------------
# Setup
# -----------------------------
device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)

# -----------------------------
# Training part
# -----------------------------
if MODE == 'train':
    from train_tracker import train_latent_epsnet_tracker

    # Construct the path to the npy file generated earlier
    DATASET_PATH = os.path.join(script_dir, "data", "x0_dataset", 
                                f"x0_{FREQ_GHZ}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_SAMPLES}samples_rT{R_T}.npy")

    print(f'[Info] Loading dataset from:\n  {DATASET_PATH}')
    x0_complex = np.load(DATASET_PATH) # Expected shape: (1000000, 5)

    # Separate real and imaginary components.
    # The new shape will be (1000000, 10), acting as the raw features for the MLP
    x0_real = np.concatenate([np.real(x0_complex), np.imag(x0_complex)], axis=-1)
    
    # Push data to PyTorch Tensor format
    x0_tensor = torch.tensor(x0_real, dtype=torch.float32)

    print(f'[Info] Input feature dimensions: {x0_tensor.shape[1]}')
    print('[Info] Training tracking epsilon net...')

    eps_net = train_latent_epsnet_tracker(
        Xs_real=x0_tensor,
        model_type=MODEL_TYPE,
        num_epochs=NUM_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=LR,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        T=T_DIFFUSION,
        val_split=VAL_SPLIT,
        patience=PATIENCE,
        device=device,
        script_dir=script_dir,
        model_file_name=MODEL_WEIGHT_FILE_NAME
    )

elif MODE == 'test':
    print("[Info] Tracking Inference Stage (To Be Implemented).")