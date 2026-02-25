import torch
import os
import numpy as np

# -----------------------------
# Configurations
# -----------------------------
RUN_ID = 2
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


# --- Tunable Tracking Parameters ---
NUM_TEST_SAMPLES = 3000
NUM_SAMPLING_STEPS = 50
# K_START defines how much noise to add to the KF prediction. 
# For rho=0.995 (very accurate prediction), a small value (15) is perfect. 
# If tracking faster users (e.g. rho=0.8), you would increase this.
K_START = 3
GUIDANCE_LAMBDA = 0.1
# -----------------------------------

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
    import scipy.io as sio
    from data.data_loader_tracker import get_tracking_testing_dataset
    from diffusion.ddim_sampler_tracker import ddim_tracking_sampler
    from models.epsnet_mlp import EpsNetMLP
    
    print("\n[Info] Initializing Tracking Inference Stage...")
    
    # 1. Load Trained Model
    weights_path = os.path.join(script_dir, "weights", MODEL_WEIGHT_FILE_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[Error] Checkpoint not found: {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location=device)
    
    dim = R_T * 2
    eps_net = EpsNetMLP(dim=dim, hidden=512, time_emb_dim=128).to(device)
    eps_net.load_state_dict(checkpoint['model_state_dict'])
    eps_net.eval()
    
    data_mean = checkpoint['data_mean'].to(device)
    data_std = checkpoint['data_std'].to(device)
    
    # 2. Load Testing Data
    # Ensure NUM_TEST_SAMPLES = 3000 at the top of your file
    dataset = get_tracking_testing_dataset(script_dir, NUM_TEST_SAMPLES)
    config = dataset["config"]
    rho = config["rho"]
    
    x0_tau = dataset["x0_tau"].to(device)
    x0_tau_plus_1 = dataset["x0_tau_plus_1"].to(device)
    M_matrix = dataset["M"].to(device)
    observations = dataset["observations"]
    
    # Calculate base signal power (needed for SNR conversions inside sampler)
    y_clean = torch.matmul(x0_tau_plus_1, M_matrix.t())
    sig_power = torch.mean(torch.abs(y_clean)**2).item()
    
    
    nmse_results = []
    
    for snr in config["snr_levels"]:
        print(f"\n--- Processing SNR = {snr} dB ---")
        y_obs = observations[snr].to(device)
        
        # 3. Denoise / Track
        x0_est = ddim_tracking_sampler(
            y_obs_complex=y_obs, 
            M_complex=M_matrix, 
            x0_tau_complex=x0_tau, 
            rho=rho,
            eps_net=eps_net, 
            data_mean=data_mean, 
            data_std=data_std, 
            snr_db=snr, 
            sig_power=sig_power,
            num_steps=NUM_SAMPLING_STEPS, 
            K_start=K_START, 
            T_DIFFUSION=T_DIFFUSION,
            beta_min=BETA_MIN, 
            beta_max=BETA_MAX, 
            guidance_lambda=GUIDANCE_LAMBDA, 
            device=device
        )
        
        # 4. Calculate Tracking NMSE
        mse = torch.mean(torch.norm(x0_tau_plus_1 - x0_est, dim=1)**2)
        ref = torch.mean(torch.norm(x0_tau_plus_1, dim=1)**2)
        nmse_db = 10 * torch.log10(mse / ref).item()
        nmse_results.append(nmse_db)
        
        print(f"  SNR {snr:2d} dB | DDIM Tracking NMSE: {nmse_db:6.2f} dB")
        
    # 5. Save Results
    res_filename = f"NMSE_Tracker_DDIM_{FREQ_GHZ}GHz.mat"
    res_path = os.path.join(script_dir, "test_results", "NMSE_raw_mats")
    os.makedirs(res_path, exist_ok=True)
    
    sio.savemat(os.path.join(res_path, res_filename), {
        'snr_range': np.array(config["snr_levels"]),
        'x0_nmse': np.array(nmse_results)
    })
    
    print(f"\n[Success] Tracking Results saved to {res_filename}")