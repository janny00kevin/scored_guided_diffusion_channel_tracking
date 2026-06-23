import torch
from diffusion.continuous_beta import alpha_bar_of_t

def complex_to_real_concat(x_cplx):
    return torch.cat([x_cplx.real, x_cplx.imag], dim=-1)

def real_to_complex_concat(x_real):
    dim = x_real.shape[-1] // 2
    return x_real[..., :dim] + 1j * x_real[..., dim:]

def ddim_tracking_sampler(y_obs_complex, M_complex, x0_tau_complex, rho,
                          eps_net, data_mean, physical_mean_complex, data_std, snr_db, sig_power,
                          num_steps=50, K_start=15, T_DIFFUSION=50.0,
                          beta_min=1e-4, beta_max=0.02, guidance_lambda=0.1, dynamic_eta=None,
                          device=None):
    device = device or y_obs_complex.device
    eps_net.eval()
    
    B = y_obs_complex.shape[0]
    # Calculate noise variance corresponding to current SNR
    sigma_n2 = sig_power * (10 ** (-snr_db / 10.0))
    max_thred = 0.01
    
    # 1. Physical Prediction
    physical_mean_complex = 0
    x_pred_cplx = rho * (x0_tau_complex - physical_mean_complex) + physical_mean_complex
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
            grad_cplx = torch.matmul(err_cplx, M_complex.conj()) / max(sigma_n2, max_thred)
            
            # Map complex gradient back to normalized real space (Chain Rule)
            def complex_to_real_matrix(M):
                Mr = M.real
                Mi = M.imag
                top = torch.cat([Mr, -Mi], dim=1)
                bot = torch.cat([Mi,  Mr], dim=1)
                return torch.cat([top, bot], dim=0)


            grad_real = complex_to_real_concat(grad_cplx)

            # Gradient with respect to normalized variable u:
            # x_real = data_mean + data_std * u
            # therefore grad_u = data_std * grad_x
            grad_norm = grad_real * data_std

            # Build effective real measurement matrix in normalized coordinates.
            # This is the matrix that DDIM actually feels.
            M_real = complex_to_real_matrix(M_complex)
            std_vec = data_std.flatten()

            M_eff = M_real * std_vec[None, :]

            sigma_eff = max(float(sigma_n2), 0.01)

            # Diagonal Hessian approximation of the negative log-likelihood:
            # H ≈ M_eff^T M_eff / sigma_eff
            diag_H = torch.sum(M_eff.abs() ** 2, dim=0) / sigma_eff
            diag_H = torch.clamp(diag_H, min=1e-8)

            # Damping prevents excessive updates in weakly observed coordinates.
            # Start with 0.1, then try 0.03 and 0.3.
            damping = 0.1
            damp = damping * torch.mean(diag_H)

            # Jacobi-preconditioned likelihood gradient.
            grad_precond = grad_norm / (diag_H[None, :] + damp)

            # Normalize average scale so eta remains interpretable.
            scale_restore = torch.mean(diag_H + damp)
            grad_precond = grad_precond * scale_restore

            if dynamic_eta is True:
                guidance_step = guidance_lambda * sqrt_1m_a_cur * grad_precond
            else:
                guidance_step = guidance_lambda * grad_precond

            x0_hat_guided_norm = x0_hat_norm + guidance_step
            
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
        grad_cplx = torch.matmul(err_cplx, M_complex.conj()) / max(sigma_n2, max_thred)
        grad_norm = complex_to_real_concat(grad_cplx) * data_std
        
        if dynamic_eta is True:
            x0_hat_final_guided_norm = x0_hat_final_norm + guidance_lambda * sqrt_1m_a_cur * grad_norm
        elif dynamic_eta is False:
            x0_hat_final_guided_norm = x0_hat_final_norm + guidance_lambda * grad_norm

        # Output final physical complex state
        x0_final_phys_real = x0_hat_final_guided_norm * data_std + data_mean
        x0_final_cplx = real_to_complex_concat(x0_final_phys_real)
        
    return x0_final_cplx