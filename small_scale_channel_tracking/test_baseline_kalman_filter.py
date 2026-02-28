import torch
import numpy as np
import os
import scipy.io

# ==========================================
# 1. Configuration
# ==========================================
NUM_TEST_SAMPLES = 3000
SNR_LEVELS = [-4, -2, 0, 2, 4, 6, 8, 10]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust to point to the newly generated dataset
DATASET_PATH = os.path.join(SCRIPT_DIR, "data", "testing_dataset", f"tracking_test_data_{NUM_TEST_SAMPLES}samples.pt")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_results", "NMSE_raw_mats")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "NMSE_Baseline_Kalman_Filter.mat")

# ==========================================
# 2. Kalman Filter Implementation
# ==========================================
def run_kalman_filter_baseline():
    print("--- Starting Kalman Filter Baseline Tracking Test ---")
    
    # 1. Load Common Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run generate_tracking_test_dataset.py first.")
        
    print(f"Loading tracking dataset: {os.path.basename(DATASET_PATH)}")
    dataset = torch.load(DATASET_PATH, map_location='cpu')
    
    config = dataset["config"]
    rho = config["rho"]
    Q_cov = torch.diag(config["process_noise_var"]).numpy()
    
    # Convert tensors to numpy for traditional KF math
    x0_tau = dataset["x0_tau"].numpy()
    x0_tau_plus_1 = dataset["x0_tau_plus_1"].numpy()
    M = dataset["M"].numpy()
    y_observations = dataset["observations"]
    
    num_pilots = config["num_pilots"]
    
    # Base signal power to calculate noise variance for R matrix
    y_clean = x0_tau_plus_1 @ M.T
    sig_power = np.mean(np.abs(y_clean)**2)
    
    nmse_results = []
    print("\nStarting KF Tracking Evaluation:")
    
    for snr in SNR_LEVELS:
        y_obs = y_observations[snr].numpy()
        
        # Determine the noise variance R for the Update step
        sigma_n2 = sig_power * (10 ** (-snr / 10.0))
        R_cov = sigma_n2 * np.eye(num_pilots)
        
        # --- KALMAN FILTER (1-Step Tracking) ---
        # A) Predict Step
        x_pred = rho * x0_tau  # A * x0(tau)
        P_pred = Q_cov         # Covariance prediction (assuming P_tau = 0 for 1-step ideal start)
        
        # B) Update Step
        # Innovation covariance: S = M * P_pred * M^H + R
        S = M @ P_pred @ M.conj().T + R_cov
        
        # Kalman Gain: K = P_pred * M^H * S^-1
        K = P_pred @ M.conj().T @ np.linalg.inv(S)
        
        # Innovation: v = y_obs - M * x_pred
        innovation = y_obs - (x_pred @ M.T)
        
        # Updated state estimate: x_hat = x_pred + K * v
        x_hat = x_pred + (innovation @ K.T)
        
        # 3. Evaluate NMSE
        mse = np.mean(np.linalg.norm(x0_tau_plus_1 - x_hat, axis=1)**2)
        ref = np.mean(np.linalg.norm(x0_tau_plus_1, axis=1)**2)
        nmse_db = 10 * np.log10(mse / ref)
        
        nmse_results.append(nmse_db)
        print(f"  SNR {snr:2d} dB | KF Tracking NMSE: {nmse_db:6.2f} dB")
        
    # 4. Save Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scipy.io.savemat(OUTPUT_FILE, {
        'snr_range': np.array(SNR_LEVELS),
        'x0_nmse': np.array(nmse_results)
    })
    print(f"\n[Info] Results saved to test_results/NMSE_raw_mats/{os.path.basename(OUTPUT_FILE)}")

if __name__ == "__main__":
    run_kalman_filter_baseline()