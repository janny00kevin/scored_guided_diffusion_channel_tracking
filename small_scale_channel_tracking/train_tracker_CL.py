import torch
import torch.nn.functional as F
import os
import copy
from diffusion.continuous_beta import alpha_bar_of_t


def info_nce_loss(features_a, features_b, temperature=0.1):
    """
    Calculates InfoNCE loss.
    features_a: (Batch, dim) - Embeddings from View A
    features_b: (Batch, dim) - Embeddings from View B
    """
    # Normalize features (Cosine Similarity)
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)

    # Similarity matrix: (Batch, Batch)
    # logits[i, j] = similarity between sample i (view A) and sample j (view B)
    # Diagonal = positive pairs from the same sample
    # Off-diagonal = in-batch negative pairs from different samples
    logits = torch.matmul(features_a, features_b.T) / temperature

    # Labels: The positive pair for i is i
    labels = torch.arange(logits.shape[0], device=logits.device)

    return F.cross_entropy(logits, labels)

def train_latent_epsnet_tracker(Xs_real, model_type='mlp', num_epochs=100, batch_size=4096, lr=1e-3,
                                beta_min=1e-4, beta_max=0.02, T=50.0,
                                val_split=0.1, patience=15,
                                device=None, script_dir=None, model_file_name=None,
                                contrastive_weight=0.0, temperature=0.1):
    
    device = device or torch.device('cpu')

    num_total = Xs_real.shape[0]
    dim = Xs_real.shape[1]

    # --- 1. Split Train and Validation ---
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    # Randomly shuffle data indices
    indices = torch.randperm(num_total, device=Xs_real.device)
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    train_data = Xs_real[train_idx].to(device)
    val_data = Xs_real[val_idx].to(device)

    # --- 2. Compute Normalization Statistics (on Training Data only) ---
    data_mean = torch.mean(train_data, dim=0, keepdim=True)
    data_std = torch.std(train_data, dim=0, keepdim=True)
    
    # Avoid division by zero for any constant features
    data_std[data_std < 1e-8] = 1.0

    # Normalize data
    train_data_norm = (train_data - data_mean) / data_std
    val_data_norm = (val_data - data_mean) / data_std

    print(f"[Info] Training Samples: {num_train} | Validation Samples: {num_val}")

    # --- 3. Model Setup ---
    if model_type == 'mlp':
        from models.epsnet_mlp import EpsNetMLP as Net
        # 512 hidden size is generally plenty for a small dimension like 10
        net = Net(dim=dim, hidden=512, time_emb_dim=128).to(device)
    else:
        raise NotImplementedError("Only 'mlp' is recommended for flat modal vectors.")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=20, verbose=False
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    best_epoch = 0

    iters_per_epoch = max(1, num_train // batch_size)
    val_iters = max(1, num_val // (batch_size * 2)) # Double batch size for eval

    # --- 4. Training Loop ---
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0.0
        total_mse = 0.0
        total_nce = 0.0

        # Shuffle training data every epoch
        perm = torch.randperm(num_train, device=device)
        train_data_shuffled = train_data_norm[perm]

        for i in range(iters_per_epoch):
            start = i * batch_size
            end = start + batch_size
            x0_batch = train_data_shuffled[start:end]
            current_bs = x0_batch.shape[0]


            perturbation_scale = 0.01
            # View 1: original sample
            x0_view1 = x0_batch

            # View 2: same sample with small perturbation
            x0_view2 = ( x0_batch + perturbation_scale * torch.randn_like(x0_batch) )

            # Sample t (Same t for both views is standard for contrastive diffusion)
            t_cont = torch.rand(current_bs, device=device) * T

            # Compute noise schedule
            a_bar = alpha_bar_of_t(t_cont, beta_min, beta_max, T).view(-1, 1)
            sqrt_a = torch.sqrt(a_bar)
            sqrt_1ma = torch.sqrt(1.0 - a_bar)

            # Add Gaussian Noise
            eps_1 = torch.randn_like(x0_view1)
            eps_2 = torch.randn_like(x0_view2)

            x_t_1 = sqrt_a * x0_view1 + sqrt_1ma * eps_1
            x_t_2 = sqrt_a * x0_view2 + sqrt_1ma * eps_2

            # Forward Pass (Get Embedding)
            pred_eps_1, z_1 = net(x_t_1, t_cont, return_embedding=True)
            pred_eps_2, z_2 = net(x_t_2, t_cont, return_embedding=True)

            # Calculate Losses
            # 1. MSE (Reconstruction) - Average of both views
            loss_mse = 0.5 * (torch.mean((pred_eps_1 - eps_1)**2) + torch.mean((pred_eps_2 - eps_2)**2))

            # 2. InfoNCE (Contrastive)
            loss_nce = info_nce_loss(z_1, z_2, temperature)

            # Total
            loss = loss_mse + contrastive_weight * loss_nce

            opt.zero_grad(); loss.backward(); opt.step()

            total_train_loss += loss.item()
            total_mse += loss_mse.item()
            total_nce += loss_nce.item()

        avg_train_loss = total_train_loss / iters_per_epoch
        avg_mse = total_mse / iters_per_epoch
        avg_nce = total_nce / iters_per_epoch

        # --- 5. Validation Step ---
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            # Standard validation on flattened snapshots (like original)
            val_iters = max(1, num_val // (batch_size * 2)) # Larger batch for eval
            
            for i in range(val_iters):
                start = i * (batch_size * 2)
                end = min(start + (batch_size * 2), num_val)
                x0_val = val_data_norm[start:end]
                current_bs = x0_val.shape[0]

                t_cont_val = torch.rand(current_bs, device=device) * T
                a_bar_val = alpha_bar_of_t(t_cont_val, beta_min, beta_max, T).view(-1, 1)

                noise_val = torch.randn_like(x0_val)
                x_t_val = torch.sqrt(a_bar_val) * x0_val + torch.sqrt(1.0 - a_bar_val) * noise_val

                pred_val = net(x_t_val, t_cont_val, return_embedding=False)
                val_loss = torch.mean((pred_val - noise_val)**2)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / val_iters

        # --- 6. Logging & Updates ---
        current_lr = opt.param_groups[0]['lr']
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_train_loss:.5f} (MSE:{avg_mse:.4f} NCE:{avg_nce:.4f}) | Val: {avg_val_loss:.5f} | LR: {current_lr:.2e} | Best Val: {best_val_loss:.5f}")
            # best = False
        scheduler.step(avg_val_loss)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience and current_lr < 1e-6:
                print(f"[Info] Early stopping triggered at epoch {epoch+1}")
                break

    # --- 7. Finalize and Save Checkpoint ---
    print(f"Training finished. Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        if script_dir:
            weights_dir = os.path.join(script_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            save_path = os.path.join(weights_dir, model_file_name)

            # Package stats into checkpoint for testing inference
            checkpoint = {
                'model_state_dict': best_model_state,
                'config': {'T': T, 'beta_min': beta_min, 'beta_max': beta_max},
                'data_mean': data_mean.cpu(),
                'data_std': data_std.cpu(),
                'epoch': best_epoch,
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved to {save_path}")

    return net