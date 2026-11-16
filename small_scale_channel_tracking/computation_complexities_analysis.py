def compute_detailed_flops(K, r_T, r_R, d_mlp, T, N_s, 
                           num_samples=1000000, val_split=0.1, batch_size=4096, total_epochs=741):
    # Calculate common structural dimensions
    r_TR = r_T * r_R
    T_Ns = T * N_s
    
    # =========================================================================
    # 1. ONLINE INFERENCE PHASE (Per Tracking Step \tau -> \tau+1)
    # =========================================================================
    
    # Kalman Filter Inference FLOPs (Dense transition matrix A)
    kf_inference_flops = (
        16 * r_TR**3 + 
        (8/3) * T_Ns**3 + 
        16 * (T_Ns**2) * r_TR + 
        8 * T_Ns * (r_TR**2) + 
        10 * r_TR**2 + 
        16 * T_Ns * r_TR +
        4 * T_Ns +
        2 * r_TR
    )
    
    # DDIM Inference FLOPs (Pure MLP execution + Likelihood Guidance over K steps)
    ddim_inference_flops = K * (
        8 * r_TR * d_mlp + 
        2 * d_mlp**2 + 
        16 * T_Ns * r_TR + 
        2 * T_Ns + 
        2 * r_TR
    )
    
    # =========================================================================
    # 2. OFFLINE TRAINING PHASE (Total Across 741 Epochs)
    # =========================================================================
    
    # Split dataset sizes
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    
    # Find exact samples processed per epoch based on integer batch drops
    iters_per_epoch = num_train // batch_size
    val_iters = num_val // (batch_size * 2)
    
    samples_trained_per_epoch = iters_per_epoch * batch_size
    samples_validated_per_epoch = val_iters * (batch_size * 2)
    
    # Core MLP Forward Pass Cost per sample
    mlp_forward_flops_per_sample = 2 * d_mlp**2 + 8 * r_TR * d_mlp
    
    # Backpropagation pass scales the training forward pass by 3x (1x Forward + 2x Backward)
    flops_per_training_sample = 3 * mlp_forward_flops_per_sample
    flops_per_validation_sample = 1 * mlp_forward_flops_per_sample # Forward pass only
    
    # Sum up total training phase workload
    training_flops_per_epoch = (
        (samples_trained_per_epoch * flops_per_training_sample) + 
        (samples_validated_per_epoch * flops_per_validation_sample)
    )
    ddim_total_training_flops = training_flops_per_epoch * total_epochs
    
    return kf_inference_flops, ddim_inference_flops, ddim_total_training_flops

# =========================================================================
# Execution Configuration
# =========================================================================
K_val = 30
r_T_val = 64
r_R_val = 1
d_mlp_val = 512
T_val = 64
N_s_val = 1

kf_inf, ddim_inf, ddim_train = compute_detailed_flops(
    K=K_val, r_T=r_T_val, r_R=r_R_val, d_mlp=d_mlp_val, T=T_val, N_s=N_s_val,
    num_samples=1000000, val_split=0.1, batch_size=4096, total_epochs=741
)

print("="*65)
print("             TRACKING INFRASTRUCTURE COMPONENT FLOPS          ")
print("="*65)
print(f"ONLINE TRACKING (Per Step):")
print(f"  -> Kalman Filter Baseline : {kf_inf / 1e6:.2f} MegaFLOPs")
print(f"  -> DDIM GenAI Tracker     : {ddim_inf / 1e6:.2f} MegaFLOPs")
print("-"*65)
print(f"OFFLINE LEARNING (Total over 741 Epochs):")
print(f"  -> DDIM Total Training    : {ddim_train / 1e12:.2f} TeraFLOPs (10^12)")
print("="*65)