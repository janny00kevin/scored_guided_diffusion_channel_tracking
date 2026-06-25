import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
from matplotlib.patches import Ellipse

# =============================
# 1. Configuration
# =============================
FREQ_GHZ = 38
if FREQ_GHZ == 12: RHO = 0.1034
elif FREQ_GHZ == 38: RHO = 0.0001
# NUM_PILOTS = 64
R_T = 64
mode = 'modal' # 'spatial' or 'modal'

# Load .mat files to plot
if mode == 'spatial':
    FILES = {
        f"Kalman Filter (T=64)": f"NMSE_KF_spatial_T64_{FREQ_GHZ}GHz_rho{RHO:.3f}.mat",
        r"DDIM Tracker w/ varied $\eta_k$ (T=64)": f"NMSE_DDIM_spatial_T64_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat",
        f"Kalman Filter (T=128)": f"NMSE_KF_spatial_T128_{FREQ_GHZ}GHz_rho{RHO:.3f}.mat",
        r"DDIM Tracker w/ varied $\eta_k$ (T=128)": f"NMSE_DDIM_spatial_T128_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat",
        f"Kalman Filter (T=320)": f"NMSE_KF_spatial_T320_{FREQ_GHZ}GHz_rho{RHO:.3f}.mat",
        r"DDIM Tracker w/ varied $\eta_k$ (T=320)": f"NMSE_DDIM_spatial_T320_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat",
    }
elif mode == 'modal':
    FILES = {
        f"Kalman Filter (T=64)": f"NMSE_KF_rT{R_T}_T64_{FREQ_GHZ}GHz_rho{RHO:.3f}.mat",
        r"DDIM Tracker w/ varied $\eta_k$ (T=64)": f"NMSE_DDIM_rT{R_T}_T64_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat",
        f"Kalman Filter (T=128)": f"NMSE_KF_rT{R_T}_T128_{FREQ_GHZ}GHz_rho{RHO:.3f}.mat",
        r"DDIM Tracker w/ varied $\eta_k$ (T=128)": f"NMSE_DDIM_rT{R_T}_T128_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat",
        f"Kalman Filter (T=320)": f"NMSE_KF_rT{R_T}_T320_{FREQ_GHZ}GHz_rho{RHO:.3f}.mat", 
        r"DDIM Tracker w/ varied $\eta_k$ (T=320)": f"NMSE_DDIM_rT{R_T}_T320_{FREQ_GHZ}GHz_rho{RHO:.3f}_pg.mat",
    }

STYLES = {
    "Kalman Filter (T=64)": {
        'color': 'blue', 
        'marker': 's', 
        'linestyle': '-', 
        'linewidth': 2, 
        'markersize': 8
    },
    # r"DDIM Tracker w/ fixed $\eta_k$ ($r_T=64$)": {
    #     'color': 'orange', 
    #     'marker': 'o', 
    #     'linestyle': '-',  
    #     'linewidth': 2, 
    #     'markersize': 8
    # },
    r"DDIM Tracker w/ varied $\eta_k$ (T=64)": {
        'color': 'green', 
        'marker': 'D', 
        'linestyle': '-',  
        'linewidth': 2, 
        'markersize': 8
    },
    "Kalman Filter (T=128)": {
        'color': 'blue', 
        'marker': 's', 
        'linestyle': '--', 
        'linewidth': 2, 
        'markersize': 8
    },
    # r"DDIM Tracker w/ fixed $\eta_k$ ($r_T=38$)": {
    #     'color': 'orange', 
    #     'marker': 'o', 
    #     'linestyle': '--',  
    #     'linewidth': 2, 
    #     'markersize': 8
    # },
    r"DDIM Tracker w/ varied $\eta_k$ (T=128)": {
        'color': 'green', 
        'marker': 'D', 
        'linestyle': '--',  
        'linewidth': 2, 
        'markersize': 8
    },
    "Kalman Filter (T=320)": {
        'color': 'blue', 
        'marker': 's', 
        'linestyle': ':', 
        'linewidth': 2, 
        'markersize': 8
    },
    # r"DDIM Tracker w/ fixed $\eta_k$ (spatial)": {
    #     'color': 'orange', 
    #     'marker': 'o', 
    #     'linestyle': ':',  
    #     'linewidth': 2, 
    #     'markersize': 8
    # },
    r"DDIM Tracker w/ varied $\eta_k$ (T=320)": {
        'color': 'green', 
        'marker': 'D', 
        'linestyle': ':',  
        'linewidth': 2, 
        'markersize': 8
    },
}

# .mat keys mapping (Tracker only focuses on x0 for now)
MAT_KEYS = {
    "snr": "snr_range",
    "x0":  "x0_nmse"
}

# =============================
# 2. Helper Functions
# =============================
def load_data(filepath):
    """Load .mat file"""
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return None
    try:
        return sio.loadmat(filepath)
    except Exception as e:
        print(f"[Error] Failed to load {filepath}: {e}")
        return None

def plot_and_save_metric(metric_type, title, ylabel, save_name_base, script_dir):
    """
    Plot and save .mat data for a single metric type.
    """
    fig, ax = plt.subplots(figsize=(8, 6)) # Changed to use ax for patch drawing
    data_found = False

    # For saving data for .mat export
    plot_data_export = {}
    
    # Store end points to draw ellipses around them later
    # Format: { 'T_value': [ (x1, y1), (x2, y2) ] }
    group_points = {'64': [], '128': [], '320': []}
    target_snr = 10  # The SNR value where we will draw the circle/ellipse
    
    # get corresponding data key
    data_key = MAT_KEYS[metric_type]
    
    # read and plot each file
    for label, filename in FILES.items():
        filepath = os.path.join(script_dir, "NMSE_raw_mats", filename)
        data = load_data(filepath)
        
        if data is None:
            continue
            
        if data_key not in data or MAT_KEYS["snr"] not in data:
            print(f"[Warning] Key '{data_key}' not found in {filename}")
            continue

        # flatten arrays
        snrs = data[MAT_KEYS["snr"]].flatten()
        values = data[data_key].flatten()
        
        # plot
        data_found = True
        style = STYLES.get(label, {'linestyle': '-', 'marker': 'o'}) 
        legend_label = style.pop('label', label)
        ax.plot(snrs, values, label=legend_label, **style)

        # ---------------------------------------------------------
        # NEW: Find the y-value at our target SNR for grouping
        # ---------------------------------------------------------
        try:
            idx = np.where(snrs == target_snr)[0][0]
            y_val = values[idx]
            
            # Determine which T group this belongs to based on the label
            if "T=64" in label:
                group_points['64'].append(y_val)
            elif "T=128" in label:
                group_points['128'].append(y_val)
            elif "T=320" in label:
                group_points['320'].append(y_val)
        except IndexError:
            pass # target_snr not found in this array

        # prepare data for .mat export
        safe_label = label.replace(" ", "_").replace("-", "_")
        plot_data_export[f"{safe_label}_snr"] = snrs
        plot_data_export[f"{safe_label}_{metric_type}"] = values
        
        # use the same x-ticks (SNR dB) for all plots
        last_valid_snrs = snrs

    if not data_found:
        print(f"[Info] No valid data found for {title}. Skipping plot.")
        plt.close()
        return

    # ---------------------------------------------------------
    # NEW: Draw Ellipses and Add Text Labels
    # ---------------------------------------------------------
    # Define colors for the groups
    group_colors = {'64': 'red', '128': 'red', '320': 'red'}
    
    for t_val, y_vals in group_points.items():
        if len(y_vals) >= 2: # We need both KF and DDIM to draw a circle around them
            # Calculate the center of the ellipse
            y_center = np.mean(y_vals)
            # Calculate the height of the ellipse to cover both points
            y_diff = abs(max(y_vals) - min(y_vals))
            
            # Add some padding to width and height so it looks like a nice circle/ellipse
            ellipse_width = 1.5   # Spread across the x-axis
            ellipse_height = y_diff + 0.8  # Spread across the y-axis
            
            # Draw the Ellipse
            ellipse = Ellipse((target_snr, y_center), width=ellipse_width, height=ellipse_height,
                              fill=False, edgecolor=group_colors[t_val], linestyle='-', linewidth=1.5, alpha=0.8)
            ax.add_patch(ellipse)
            
            # Add the "T=X" Text label right next to the ellipse
            ax.text(target_snr + 1.0, y_center, f"T={t_val}", color=group_colors[t_val], 
                    fontsize=12, fontweight='bold', va='center')


    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Put legend in the lower left so it doesn't overlap with the new circles
    ax.legend(fontsize=12, loc='lower left') 
    
    if 'last_valid_snrs' in locals():
        # Extend x-axis slightly so the circles and text aren't cut off at the edge
        ax.set_xlim([min(last_valid_snrs)-0.5, max(last_valid_snrs)+2.5])
        ax.set_xticks(last_valid_snrs)
        
    plt.tight_layout()
    
    # Make sure output directories exist
    os.makedirs(os.path.join(script_dir, "NMSE_plot_png"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "NMSE_plot_mat"), exist_ok=True)
    
    # save png plot
    png_path = os.path.join(script_dir, f"NMSE_plot_png/{save_name_base}.png")
    plt.savefig(png_path, dpi=300)
    print(f"[Success] Saved plot image to: NMSE_plot_png/{save_name_base}.png")

    # save .mat data
    mat_path = os.path.join(script_dir, f"NMSE_plot_mat/{save_name_base}.mat")
    sio.savemat(mat_path, plot_data_export)
    print(f"[Success] Saved plot data to : NMSE_plot_mat/{save_name_base}.mat")
    plt.close()

# =============================
# 3. Main Execution
# =============================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("[Info] Generating tracking plots...")

    if mode == 'spatial':
        save_name_base = f"NMSE_Tracker_{FREQ_GHZ}GHz_spatial_rho{RHO:.3f}_T64vs128vs320_pg"
    elif mode == 'modal':
        save_name_base = f"NMSE_Tracker_{FREQ_GHZ}GHz_rT{R_T}_rho{RHO:.3f}_T64vs128vs320_pg"

    # Plot Tracking NMSE (x0 state)
    plot_and_save_metric(
        metric_type="x0",
        title="Channel Tracking Performance",
        ylabel="Tracking NMSE (dB)",
        save_name_base=save_name_base,
        script_dir=script_dir
    )

if __name__ == "__main__":
    main()