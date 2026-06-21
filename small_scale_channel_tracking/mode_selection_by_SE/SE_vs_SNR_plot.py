import os
import scipy.io as sio
import matplotlib.pyplot as plt

# --- 1. Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, "SE_vs_SNR_result")

# Match the parameters from your main simulation
FREQ_GHZ = 12
RT_LIST = [64, 60]

# Construct the exact filename that was saved by the main script
mat_filename = f"SE_vs_SNR_{FREQ_GHZ}GHz_rT{RT_LIST}.mat"
mat_filepath = os.path.join(result_dir, mat_filename)

if not os.path.exists(mat_filepath):
    print(f"Error: Could not find the data file at {mat_filepath}")
    print("Please make sure you have run SE_vs_SNR.py first.")
    exit()

# --- 2. Load Data ---
print(f"Loading data from {mat_filename}...")
data = sio.loadmat(mat_filepath)

# Extract and flatten the arrays 
# (scipy.io.loadmat loads arrays as 2D matrices by default, so .flatten() makes them 1D for plotting)
SNR_DB_RANGE = data['SNR_DB_RANGE'].flatten()
result_port = data['result_port'].flatten()

results_modal_dict = {}
for rt in RT_LIST:
    key_name = f'rate_modal_{rt}'
    results_modal_dict[rt] = data[key_name].flatten()

# --- 3. Plotting: Rate Comparison Curves ---
print("Generating Rate Comparison Curves...")
# plt.figure(figsize=(10, 6))
plt.plot(SNR_DB_RANGE, result_port, 'k-o', linewidth=2.5, label='Port-domain Full-Digital Bound')

markers = ['s', '^', 'v', 'd', 'p', '*'] 
for idx, rt in enumerate(RT_LIST):
    m = markers[idx % len(markers)]
    plt.plot(SNR_DB_RANGE, results_modal_dict[rt], f'--{m}', label=rf'Modal-domain ($r_T$={rt})')

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Average Rate (bps/Hz)', fontsize=12)
plt.title(f'Average Sum Rate Comparison ({FREQ_GHZ}GHz Imperfect CSI)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')

plot1_path = os.path.join(result_dir, f'SE_vs_SNR_{FREQ_GHZ}GHz_curves.png')
plt.savefig(plot1_path, bbox_inches='tight', dpi=300)
plt.close() # Close the figure to free up memory

# --- 4. Plotting: Modal Efficiency Percentage Table ---
print("Generating Modal Efficiency Table...")
table_data = []
for snr_idx in range(len(SNR_DB_RANGE)):
    row_percent = []
    port_val = result_port[snr_idx]
    for rt in RT_LIST:
        modal_val = results_modal_dict[rt][snr_idx]
        percent = (modal_val / port_val) * 100
        row_percent.append(f"{percent:.2f}%") 
    table_data.append(row_percent)

fig_table, ax_table = plt.subplots(figsize=(10, 6))
ax_table.axis('tight')
ax_table.axis('off') 

col_labels = [f'Modal (rT={rt})' for rt in RT_LIST]
row_labels = [f'{snr} dB' for snr in SNR_DB_RANGE]

the_table = ax_table.table(cellText=table_data,
                           rowLabels=row_labels,
                           colLabels=col_labels,
                           rowColours=["#f1f1f1"] * len(row_labels),
                           colColours=["#cfe2f3"] * len(col_labels), 
                           cellLoc='center',
                           loc='center')

the_table.auto_set_font_size(False)
the_table.set_fontsize(11)    
the_table.scale(1.2, 1.6)      

plt.title(f'Modal Efficiency Relative to Port-domain Bound (%) - {FREQ_GHZ}GHz', y=1.05, fontweight='bold', fontsize=13)
plot2_path = os.path.join(result_dir, f'SE_vs_SNR_table_{FREQ_GHZ}GHz.png')
plt.savefig(plot2_path, bbox_inches='tight', dpi=300)

print(f"\nSuccess! Both plots have been saved to:\n{result_dir}")

# Uncomment the line below if you want the plots to pop up on your screen when you run the script
# plt.show()