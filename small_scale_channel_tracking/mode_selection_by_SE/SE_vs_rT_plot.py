import os
import scipy.io as sio
import matplotlib.pyplot as plt

# --- 1. Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, "SE_vs_rT_result")

FREQ_GHZ = 38
mat_filename = f"SE_vs_rT_{FREQ_GHZ}GHz_data.mat"
mat_filepath = os.path.join(result_dir, mat_filename)

if not os.path.exists(mat_filepath):
    print(f"Error: Could not find the data file at {mat_filepath}")
    exit()

# --- 2. Load Data ---
print(f"Loading data from {mat_filename}...")
data = sio.loadmat(mat_filepath)

SNR_LIST = data['SNR_LIST'].flatten()
RT_RANGE = data['RT_RANGE'].flatten()

# --- 3. Plotting Loop for Each SNR ---
for snr in SNR_LIST:
    print(f"\nGenerating plots for SNR = {snr} dB...")
    snr_str = f"n{abs(snr)}" if snr < 0 else f"p{snr}"
    
    baseline = data[f'baseline_snr_{snr_str}'].flatten()[0]
    res_ideal = data[f'ideal_snr_{snr_str}'].flatten()
    res_hybrid_E = data[f'hybrid_snr_{snr_str}'].flatten()

    # --- A. Generate the Line Plot (IEEE Compliant) ---
    # 3.5 inches is the standard width for an IEEE single-column figure
    plt.figure(figsize=(3.5, 2.6))
    
    plt.plot(RT_RANGE, res_ideal, '-', color='b', linewidth=1.2, 
             label='Modal Fully Digital')
    plt.plot(RT_RANGE, res_hybrid_E, '--o', color='g', linewidth=1.2, markersize=3, 
             label='Modal Hybrid w/ E')

    # Plot the full-digital baseline
    plt.axhline(y=baseline, color='k', linestyle=':', linewidth=1.5, label='Port Bound')

    # IEEE requires smaller font sizes (8pt to 9pt)
    plt.xlabel('Number of Modes ($r_T$)', fontsize=9)
    plt.ylabel('Spectral Efficiency (bps/Hz)', fontsize=9)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust tick font sizes for print readability
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    # Compact legend without a bulky frame
    plt.legend(loc='lower right', fontsize=7, frameon=False)
    
    plot_path = os.path.join(result_dir, f'SE_vs_rT_curve_{FREQ_GHZ}GHz_SNR_{snr_str}dB.png')
    
    # dpi=300 is the strict IEEE minimum for publication quality
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    # --- B. Generate the Efficiency Table ---
    table_data = []
    for idx, rt in enumerate(RT_RANGE):
        val_i = res_ideal[idx]
        val_e = res_hybrid_E[idx]
        
        eff_i = (val_i / baseline) * 100
        eff_e = (val_e / baseline) * 100
        
        row = [
            f"r_T = {rt}",
            f"{val_i:.2f} ({eff_i:.1f}%)",
            f"{val_e:.2f} ({eff_e:.1f}%)"
        ]
        table_data.append(row)

    col_labels = ['Mode Config', 'Ideal Mode', 'Hybrid w/ E']
    col_colors = ["#f2f2f2", "#d9ead3", "#cfe2f3"]

    fig, ax = plt.subplots(figsize=(6, 12)) 
    ax.axis('off')

    the_table = ax.table(cellText=table_data, 
                         colLabels=col_labels, 
                         colColours=col_colors,
                         cellLoc='center', 
                         loc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    for position, cell in the_table.get_celld().items():
        cell.set_height(0.025) 
        if position[0] == 0: 
            cell.set_text_props(weight='bold')
            cell.set_height(0.035)

    plt.title(f'Efficiency Analysis (SNR={snr}dB, {FREQ_GHZ}GHz Imperfect CSI)\nBaseline: Port Domain Bound', 
              y=0.97, fontweight='bold', fontsize=12)

    table_path = os.path.join(result_dir, f'SE_vs_rT_Table_{FREQ_GHZ}GHz_SNR_{snr_str}dB.png')
    plt.savefig(table_path, bbox_inches='tight', dpi=300)
    plt.close()

print(f"\nSuccess! All IEEE plots and tables have been saved to:\n{result_dir}")