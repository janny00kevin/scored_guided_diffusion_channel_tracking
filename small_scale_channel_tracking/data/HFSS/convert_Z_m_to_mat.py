import os
import re
import numpy as np
import scipy.io as sio

# ==========================================
# Configuration
# ==========================================
FREQ_GHZ = 12.45
TX_DIM = [8, 8]

# Input / Output Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Z_result")

# This assumes your HFSS exports are named like "12GHz_patch_8_8.m"
INPUT_FILENAME = f"{int(FREQ_GHZ)}GHz_patch_{TX_DIM[0]}_{TX_DIM[1]}.m"
INPUT_FILE = os.path.join(OUTPUT_DIR, INPUT_FILENAME)

OUTPUT_FILENAME = f"{TX_DIM[0]}x{TX_DIM[1]}_UPA_{int(FREQ_GHZ)}GHz_Z.mat"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def main():
    print(f"--- Converting HFSS Z-Matrix for {FREQ_GHZ} GHz ---")
    print(f"Reading from: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # Read the raw MATLAB text file
    with open(INPUT_FILE, 'r') as f:
        content = f.read()

    # The data is inside brackets: Z(1,:,:) = [ ... ];
    # Use regex to extract everything between the outer brackets
    match = re.search(r'Z\(1,:,\:\)\s*=\s*\[(.*?)\];', content, re.DOTALL)
    if not match:
        print("Error: Could not parse Z matrix data block from file.")
        return
        
    data_str = match.group(1)
    
    # HFSS exports rows separated by semicolons
    rows_str = data_str.split(';')
    
    num_antennas = TX_DIM[0] * TX_DIM[1]
    Z_matrix = np.zeros((num_antennas, num_antennas), dtype=complex)
    
    # Parse each row
    for i, row in enumerate(rows_str):
        if not row.strip():
            continue
            
        # Extract all complex numbers like "5.794857E+01 + 1.726039E+01i"
        # Regex looks for float +/- float i
        complex_strs = re.findall(r'([+-]?\s*\d+\.\d+E[+-]\d+)\s*([+-])\s*(\d+\.\d+E[+-]\d+)i', row)
        
        for j, (real_part, sign, imag_part) in enumerate(complex_strs):
            # Clean up spaces
            real_val = float(real_part.replace(' ', ''))
            imag_val = float(imag_part.replace(' ', ''))
            
            if sign.strip() == '-':
                imag_val = -imag_val
                
            Z_matrix[i, j] = real_val + 1j * imag_val

    print(f"Successfully parsed Z matrix of shape {Z_matrix.shape}")
    
    # Save to standard .mat format
    sio.savemat(OUTPUT_FILE, {'Z_matrix': Z_matrix})
    print(f"Saved converted matrix to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()