import os
import torch

def get_tracking_testing_dataset(script_dir, num_samples, rho, freq_ghz, rT, num_pilots, chan_mode):
    dataset_dir = os.path.join(script_dir, "data", "training_testing_dataset")
    if chan_mode == 'spatial':
        filename = f"tracking_test_nondiag_spatial_T{num_pilots}_rho{rho:.3f}_{freq_ghz}GHz_{num_samples}samples.pt"
    else:
        filename = f"tracking_test_nondiag_rT{rT}_T{num_pilots}_rho{rho:.3f}_{freq_ghz}GHz_{num_samples}samples.pt"
    file_path = os.path.join(dataset_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] Dataset not found: {file_path}. Run generate_tracking_test_dataset.py first.")
    
    print(f"[Info] Loading tracking testing dataset: {filename}...")
    dataset = torch.load(file_path, map_location='cpu')
    return dataset