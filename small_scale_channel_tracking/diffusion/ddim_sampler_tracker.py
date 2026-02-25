import torch
from diffusion.continuous_beta import alpha_bar_of_t

def complex_to_real_concat(x_cplx):
    return torch.cat([x_cplx.real, x_cplx.imag], dim=-1)

def real_to_complex_concat(x_real):
    dim = x_real.shape[-1] // 2
    return x_real[..., :dim] + 1j * x_real[..., dim:]

def ddim_tracking_sampler(y_obs_complex, M_complex, x0_tau_complex, rho,
                          eps_net, data_mean, data_std, snr_db, sig_power,
                          num_steps=50, K_start=15, T_DIFFUSION=50.0,
                          beta_min=1e-4, beta_max=0.02, guidance_lambda=0.1, 
                          device=None):
    device = device or y_obs_complex.device
    eps_net.eval()
    
    B = y_obs_complex.shape[0]
    # Calculate noise variance corresponding to current SNR
    sigma_n2 = sig_power * (10 ** (-snr_db / 10.0))
    
    # 1. Physical Prediction
    x_pred_cplx = rho * x0_tau_complex
    x_pred_real = complex_to_real_concat(x_pred_cplx)
    
    # Map to normalized latent space for the neural network
    x_pred_norm = (x_pred_real - data_mean) / data_std
    
    with torch.no_grad():
        # Define discrete time steps from K_start down to 0
        t_seq = torch.linspace(T_DIFFUSION * (K_start / num_steps), 0.0, K_start, device=device)
        
        # 2. Add diffusion noise up to intermediate step K (Eq 23)
        t_K = t_seq[0]
        a_bar_K = alpha_bar_of_t(t_K, beta_min, beta_max, T_DIFFUSION)
        sqrt_a_K = torch.sqrt(a_bar_K)
        sqrt_1m_a_K = torch.sqrt(1.0 - a_bar_K)
        
        x_t = sqrt_a_K * x_pred_norm + sqrt_1m_a_K * torch.randn_like(x_pred_norm)
        
        # 3. Denoising Loop
        for k in range(K_start - 1):
            t_cur = t_seq[k]
            t_next = t_seq[k+1]
            t_batch = torch.full((B,), t_cur, device=device)
            
            # Predict noise
            eps_pred = eps_net(x_t, t_batch)
            
            a_bar_cur = alpha_bar_of_t(t_cur, beta_min, beta_max, T_DIFFUSION)
            a_bar_next = alpha_bar_of_t(t_next, beta_min, beta_max, T_DIFFUSION)
            sqrt_a_cur = torch.sqrt(a_bar_cur)
            sqrt_1m_a_cur = torch.sqrt(1.0 - a_bar_cur)
            sqrt_a_next = torch.sqrt(a_bar_next)
            sqrt_1m_a_next = torch.sqrt(1.0 - a_bar_next)
            
            # Predict clean normalized x0
            x0_hat_norm = (x_t - sqrt_1m_a_cur * eps_pred) / (sqrt_a_cur + 1e-12)
            
            # --- LIKELIHOOD GUIDANCE (Eq 26 formulation) ---
            # Un-normalize to physical complex space to evaluate likelihood
            x0_hat_phys_real = x0_hat_norm * data_std + data_mean
            x0_hat_phys_cplx = real_to_complex_concat(x0_hat_phys_real)
            
            # Gradient: M^H * (y - M * x0) / sigma^2
            err_cplx = y_obs_complex - torch.matmul(x0_hat_phys_cplx, M_complex.t())
            grad_cplx = torch.matmul(err_cplx, M_complex.conj()) / max(sigma_n2, 0.1)
            
            # Map complex gradient back to normalized real space (Chain Rule)
            grad_real = complex_to_real_concat(grad_cplx)
            grad_norm = grad_real * data_std
            
            # Apply guidance
            x0_hat_guided_norm = x0_hat_norm + guidance_lambda * grad_norm
            
            # Recalculate equivalent noise to step down properly
            eps_guided = (x_t - sqrt_a_cur * x0_hat_guided_norm) / (sqrt_1m_a_cur + 1e-12)
            
            # DDIM Step down (Eq 27)
            x_t = sqrt_a_next * x0_hat_guided_norm + sqrt_1m_a_next * eps_guided
            
        # 4. Final Step (k=0)
        t_last = t_seq[-1]
        t_batch = torch.full((B,), t_last, device=device)
        eps_final = eps_net(x_t, t_batch)
        a_bar_last = alpha_bar_of_t(t_last, beta_min, beta_max, T_DIFFUSION)
        x0_hat_final_norm = (x_t - torch.sqrt(1.0 - a_bar_last) * eps_final) / torch.sqrt(a_bar_last)
        
        # Final Guidance
        x0_hat_phys_real = x0_hat_final_norm * data_std + data_mean
        x0_hat_phys_cplx = real_to_complex_concat(x0_hat_phys_real)
        err_cplx = y_obs_complex - torch.matmul(x0_hat_phys_cplx, M_complex.t())
        grad_cplx = torch.matmul(err_cplx, M_complex.conj()) / max(sigma_n2, 0.1)
        grad_norm = complex_to_real_concat(grad_cplx) * data_std
        
        x0_hat_final_guided_norm = x0_hat_final_norm + guidance_lambda * grad_norm
        
        # Output final physical complex state
        x0_final_phys_real = x0_hat_final_guided_norm * data_std + data_mean
        x0_final_cplx = real_to_complex_concat(x0_final_phys_real)
        
    return x0_final_cplx