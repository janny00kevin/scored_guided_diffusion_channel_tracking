import torch
import os
import numpy as np

# -----------------------------
# Configurations
# -----------------------------
RUN_ID = 2
MODE = {1: 'train', 2: 'test'}.get(RUN_ID, 'train')

# Scenario Configs
FREQ_GHZ = 38
TX_DIM = [8, 8]
RX_DIM = [1, 1]
if FREQ_GHZ == 12: RHO = 0.1034
elif FREQ_GHZ == 38: RHO = 0.0001
R_T = 64
NUM_SAMPLES = 1000000
CUDA = 1
CHAN_MODE = 'modal' # 'spatial' or 'modal'
NUM_PILOTS = 64*5

# Training settings
NUM_EPOCHS = 10000
TRAIN_BATCH_SIZE = 4096 
LR = 1e-3
MODEL_TYPE = 'mlp'
VAL_SPLIT = 0.1
PATIENCE = 10

# Diffusion Process Settings
BETA_MIN = 1e-4
BETA_MAX = 0.02
T_DIFFUSION = 1000.0


# --- Tunable Tracking Parameters ---
NUM_TEST_SAMPLES = 3000
NUM_SAMPLING_STEPS = 1000
K_START = 30
DYNAMIC_ETA = True
if DYNAMIC_ETA: 
    if CHAN_MODE == 'spatial': GUIDANCE_LAMBDA = 8e-3   #eta
    elif CHAN_MODE == 'modal': 
        if FREQ_GHZ == 12: GUIDANCE_LAMBDA = 9e-3
        elif FREQ_GHZ == 38 and R_T == 54: GUIDANCE_LAMBDA = 8e-3
        elif FREQ_GHZ == 38 and R_T == 64: GUIDANCE_LAMBDA = 7e-3
            
else: GUIDANCE_LAMBDA = 1.2
# -----------------------------------
if CHAN_MODE == 'spatial':
    MODEL_WEIGHT_FILE_NAME = f"Tracker_DDIM_{FREQ_GHZ}GHz_spatial_{MODEL_TYPE}_lr{LR:.0e}.pth"
elif CHAN_MODE == 'modal':
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
    if CHAN_MODE == 'spatial':
        DATASET_PATH =  os.path.join(script_dir, "data", "training_testing_dataset", 
                                     f"x0_{int(FREQ_GHZ)}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_SAMPLES}samples_spatial.pt")
    elif CHAN_MODE == 'modal':
        DATASET_PATH = os.path.join(script_dir, "data", "training_testing_dataset", 
                                    f"x0_{int(FREQ_GHZ)}GHz_{TX_DIM[0]}x{TX_DIM[1]}Tx_{RX_DIM[0]}x{RX_DIM[1]}Rx_{NUM_SAMPLES}samples_rT{R_T}.pt")

    print(f'[Info] Loading dataset from:\n  {DATASET_PATH}')
    x0_complex = torch.load(DATASET_PATH) # Expected shape: (1000000, 64)

    # Separate real and imaginary components.
    # The new shape will be (1000000, 76), acting as the raw features for the MLP
    x0_tensor = torch.cat([x0_complex.real, x0_complex.imag], dim=-1).float()

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
    
    # print("\n[Info] Initializing Tracking Inference Stage...")

    # def regularized_dc_update(y_obs, M, x_prior, sigma_n2, dc_reg=0.1):
    #     """
    #     Regularized data-consistency update.

    #     Observation model:
    #         y = x @ M.T

    #     Solves:
    #         min_x ||y - x M.T||^2 / sigma_n2 + lambda ||x - x_prior||^2

    #     dc_reg:
    #         smaller -> closer to LS
    #         larger  -> closer to DDIM prior
    #     """
    #     D = M.shape[1]
    #     device = M.device
    #     dtype = M.dtype

    #     sigma_eff = max(float(sigma_n2), 0.01)

    #     I = torch.eye(D, dtype=dtype, device=device)

    #     gram = M.conj().t() @ M / sigma_eff

    #     gram_scale = torch.real(torch.trace(gram)) / D
    #     lam = dc_reg * gram_scale

    #     A = gram + lam * I

    #     B = M.conj().t() @ y_obs.t() / sigma_eff + lam * x_prior.t()

    #     x_dc = torch.linalg.solve(A, B).t()

    #     return x_dc

    # def nmse_db_complex(x_true, x_hat):
    #     """
    #     Compute NMSE in dB for complex-valued tensors.

    #     x_true: ground-truth complex channel, shape [N, D]
    #     x_hat : estimated complex channel, shape [N, D]

    #     NMSE = E[||x_true - x_hat||^2] / E[||x_true||^2]
    #     NMSE_dB = 10 log10(NMSE)
    #     """
    #     mse = torch.mean(torch.sum(torch.abs(x_true - x_hat) ** 2, dim=1))
    #     ref = torch.mean(torch.sum(torch.abs(x_true) ** 2, dim=1))
    #     nmse = mse / ref
    #     return 10.0 * torch.log10(nmse).item()
    
    # 1. Load Trained Model
    weights_path = os.path.join(script_dir, "weights", MODEL_WEIGHT_FILE_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[Error] Checkpoint not found: {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location=device)
    
    if CHAN_MODE == 'spatial':
        dim = 64 * 2
    elif CHAN_MODE == 'modal':
        dim = R_T * 2
    eps_net = EpsNetMLP(dim=dim, hidden=512, time_emb_dim=128).to(device)
    eps_net.load_state_dict(checkpoint['model_state_dict'])
    eps_net.eval()
    
    data_mean = checkpoint['data_mean'].to(device)        ##################
    data_std = checkpoint['data_std'].to(device)
    # data_mean = torch.zeros_like(checkpoint['data_mean']).to(device)
    # data_std = torch.ones_like(checkpoint['data_std']).to(device)
    
    # 2. Load Testing Data
    # Ensure NUM_TEST_SAMPLES = 3000 at the top of your file
    dataset = get_tracking_testing_dataset(script_dir, NUM_TEST_SAMPLES, RHO, FREQ_GHZ, R_T, NUM_PILOTS, CHAN_MODE)
    config = dataset["config"]
    rho = config["rho"]
    
    x0_tau = dataset["x0_tau"].to(device)
    x0_tau_plus_1 = dataset["x0_tau_plus_1"].to(device)
    M_matrix = dataset["M"].to(device)
    observations = dataset["observations"]
    
    # Calculate base signal power (needed for SNR conversions inside sampler)
    y_clean = torch.matmul(x0_tau_plus_1, M_matrix.t())
    sig_power = torch.mean(torch.abs(y_clean)**2).item()
    physical_mean_complex = torch.mean(x0_tau, dim=0)
    
    
    nmse_results = []
    
    for snr in config["snr_levels"]:
        # print(f"\n--- Processing SNR = {snr} dB ---")
        y_obs = observations[snr].to(device)
        
        # 3. Denoise / Track
        x0_est = ddim_tracking_sampler(
            y_obs_complex=y_obs, 
            M_complex=M_matrix, 
            x0_tau_complex=x0_tau, 
            rho=rho,
            eps_net=eps_net, 
            data_mean=data_mean, 
            physical_mean_complex = physical_mean_complex, 
            data_std=data_std, 
            snr_db=snr, 
            sig_power=sig_power,
            num_steps=NUM_SAMPLING_STEPS, 
            K_start=K_START, 
            T_DIFFUSION=T_DIFFUSION,
            beta_min=BETA_MIN, 
            beta_max=BETA_MAX, 
            guidance_lambda=GUIDANCE_LAMBDA,
            dynamic_eta=DYNAMIC_ETA, 
            device=device
        )

        # sigma_n2 = sig_power * (10 ** (-snr / 10.0))

        # if CHAN_MODE == "modal" and NUM_PILOTS == 320 and R_T == 64:
        #     for dc_reg in [0.01, 0.03, 0.1, 0.3, 1.0]:
        #         x_dc = regularized_dc_update(
        #             y_obs=y_obs,
        #             M=M_matrix,
        #             x_prior=x0_est,
        #             sigma_n2=sigma_n2,
        #             dc_reg=dc_reg
        #         )

        #         dc_nmse = nmse_db_complex(x0_tau_plus_1, x_dc)

        #         print(
        #             f"[Post-DDIM DC] SNR={snr:3d} dB | "
        #             f"dc_reg={dc_reg:.2e} | "
        #             f"DC NMSE={dc_nmse:7.2f} dB"
        #         )
        
        # 4. Calculate Tracking NMSE
        mse = torch.mean(torch.norm(x0_tau_plus_1 - x0_est, dim=1)**2)
        ref = torch.mean(torch.norm(x0_tau_plus_1, dim=1)**2)
        nmse_db = 10 * torch.log10(mse / ref).item()
        nmse_results.append(nmse_db)
        
        # print(f"  SNR {snr:2d} dB | DDIM Tracking NMSE: {nmse_db:6.2f} dB")
        
    # print("\n" + "="*60)
    # print("FINAL TRACKING NMSE RESULTS")
    print("="*60)
    
    # Format as aligned horizontal arrays
    snr_str  = " | ".join([f"{snr:6d}" for snr in config["snr_levels"]])
    nmse_str = " | ".join([f"{nmse:6.2f}" for nmse in nmse_results])
    
    print(f"SNR (dB)  : [ {snr_str} ]")
    print(f"NMSE (dB) : [ {nmse_str} ]")
    print("="*60)
    
    # 5. Save Results
    if CHAN_MODE == 'spatial':
        if DYNAMIC_ETA: res_filename = f"NMSE_DDIM_spatial_T{NUM_PILOTS}_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat"
        else: res_filename = f"NMSE_DDIM_spatial_T{NUM_PILOTS}_{FREQ_GHZ}GHz_rho{RHO:.3f}_fixed_eta.mat"
    elif CHAN_MODE == 'modal':
        if DYNAMIC_ETA: res_filename = f"NMSE_DDIM_rT{R_T}_T{NUM_PILOTS}_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat"
        else: res_filename = f"NMSE_DDIM_rT{R_T}_T{NUM_PILOTS}_{FREQ_GHZ}GHz_rho{RHO:.3f}_fixed_eta.mat"
    res_path = os.path.join(script_dir, "test_results", "NMSE_raw_mats")
    os.makedirs(res_path, exist_ok=True)
    
    sio.savemat(os.path.join(res_path, res_filename), {
        'snr_range': np.array(config["snr_levels"]),
        'x0_nmse': np.array(nmse_results)
    })
    
    # print(f"[Success] Tracking Results saved to {res_filename}")