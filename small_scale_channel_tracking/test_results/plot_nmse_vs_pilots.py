import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def load_nmse_for_snr(filepath, target_snr):
    """Loads the .mat file and extracts the NMSE for a specific SNR."""
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return None
    
    data = sio.loadmat(filepath)
    snr_range = data['snr_range'].flatten()
    nmse_vals = data['x0_nmse'].flatten()
    
    # Find the index of the target SNR
    try:
        idx = np.where(snr_range == target_snr)[0][0]
        return nmse_vals[idx]
    except IndexError:
        print(f"[Warning] SNR {target_snr} not found in {filepath}")
        return None

def get_filepath(method, T, freq=12, rho=0.103):
    """Generates the filename, checking both new and old naming conventions."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if method == 'KF':
        name1 = f"NMSE_KF_spatial_T{T}_{freq}GHz_rho{rho}.mat"
        name2 = f"NMSE_Baseline_Kalman_Filter_spatial_T{T}_{freq}GHz_rho{rho:.3f}.mat"
    elif method == 'DDIM_fixed_eta':
        # New filename format for fixed eta
        name1 = f"NMSE_DDIM_spatial_T{T}_{freq}GHz_rho{rho:.3f}_fixed_eta.mat"
        name2 = name1 
    else:
        name1 = f"NMSE_DDIM_spatial_T{T}_{freq}GHz_rho{rho}.mat"
        name2 = f"NMSE_Tracker_DDIM_spatial_T{T}_{freq}GHz_rho{rho:.3f}.mat"
        
    path1 = os.path.join(script_dir, "NMSE_raw_mats", name1)
    path2 = os.path.join(script_dir, "NMSE_raw_mats", name2)
    
    # Return path1 if it exists, otherwise fall back to path2 (useful for T=64)
    if os.path.exists(path1): return path1
    if os.path.exists(path2): return path2
    
    # If neither exists in NMSE_raw_mats, check the current directory directly
    path3 = os.path.join(script_dir, name1)
    if os.path.exists(path3): return path3
    
    return path1 # Default return if nothing is found to trigger the warning

def main():
    print("--- Generating NMSE vs. Pilots (T) Plot ---")
    
    T_values = [64, 320, 640, 960, 1280]
    
    # Initialize lists to hold the extracted NMSE values
    kf_m4, kf_p10 = [], []
    ddim_m4, ddim_p10 = [], []
    ddim_fixed_m4, ddim_fixed_p10 = [], [] # <-- ADDED
    
    # Extract data
    for T in T_values:
        kf_path = get_filepath('KF', T)
        ddim_path = get_filepath('DDIM', T)
        ddim_fixed_path = get_filepath('DDIM_fixed_eta', T) # <-- ADDED
        
        kf_m4.append(load_nmse_for_snr(kf_path, -4))
        kf_p10.append(load_nmse_for_snr(kf_path, 10))
        
        ddim_m4.append(load_nmse_for_snr(ddim_path, -4))
        ddim_p10.append(load_nmse_for_snr(ddim_path, 10))

        # <-- ADDED
        ddim_fixed_m4.append(load_nmse_for_snr(ddim_fixed_path, -4))
        ddim_fixed_p10.append(load_nmse_for_snr(ddim_fixed_path, 10))

    # --- Plotting ---
    plt.figure(figsize=(9, 6.5))
    
    # KF Curves (Red)
    plt.plot(T_values, kf_m4, 
             color='blue', marker='s', linestyle='--', linewidth=2, markersize=8, 
             label='KF (SNR = -4 dB)')
    plt.plot(T_values, kf_p10, 
             color='blue', marker='s', linestyle='-', linewidth=2, markersize=8, 
             label='KF (SNR = 10 dB)')
    
    # DDIM Curves (Green, Diamond marker)
    plt.plot(T_values, ddim_m4, 
             color='green', marker='D', linestyle='--', linewidth=2, markersize=8, 
             label='DDIM w/ varying $\eta$ (SNR = -4 dB)')
    plt.plot(T_values, ddim_p10, 
             color='green', marker='D', linestyle='-', linewidth=2, markersize=8, 
             label='DDIM w/ varying $\eta$ (SNR = 10 dB)')
    
    # # DDIM Fixed Eta Curves (Orange, Circle marker)
    # plt.plot(T_values, ddim_fixed_m4, 
    #          color='orange', marker='o', linestyle='--', linewidth=2, markersize=8, 
    #          label='DDIM Fixed $\eta$ (SNR = -4 dB)')
    # plt.plot(T_values, ddim_fixed_p10, 
    #          color='orange', marker='o', linestyle='-', linewidth=2, markersize=8, 
    #          label='DDIM Fixed $\eta$ (SNR = 10 dB)')
    
    # Formatting
    plt.xlabel('Number of Pilots ($T$)', fontsize=14, fontweight='bold')
    plt.ylabel('Tracking NMSE (dB)', fontsize=14, fontweight='bold')
    plt.title('Tracking Performance vs. Pilot Overhead (Spatial Domain)', fontsize=16, fontweight='bold')
    
    # Force X-axis ticks to only show the specific T values
    plt.xticks(T_values, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Grid and Legend
    plt.grid(True, which='both', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.legend(fontsize=12, loc='best', framealpha=0.9, edgecolor='black')
    
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'NMSE_plot_png/NMSE_vs_Pilots_spatial.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[Success] Plot saved to: {output_path}")

if __name__ == "__main__":
    main()