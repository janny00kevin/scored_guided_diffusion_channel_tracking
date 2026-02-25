import os
import torch

def get_tracking_testing_dataset(script_dir, num_samples):
    dataset_dir = os.path.join(script_dir, "data", "testing_dataset")
    filename = f"tracking_test_data_{num_samples}samples.pt"
    file_path = os.path.join(dataset_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] Dataset not found: {file_path}. Run generate_tracking_test_dataset.py first.")
    
    print(f"[Info] Loading tracking testing dataset: {filename}...")
    dataset = torch.load(file_path, map_location='cpu')
    return dataset