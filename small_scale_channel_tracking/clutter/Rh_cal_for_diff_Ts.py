import scipy.special as sp
import numpy as np

# System parameters
f_0 = 12.45e9  # Carrier Frequency: 12.45 GHz
v = 15.0       # Velocity: 15 m/s (approx. 54 km/h or 33 mph)
c = 3e8        # Speed of light in m/s

# 1. Calculate Maximum Doppler Shift (f_D)
f_D = f_0 * (v / c)

# 2. Define the 4 standard TRS Periodicities in milliseconds
periodicities_ms = [10, 20, 40, 80]

print(f"Carrier Frequency: {f_0 / 1e9} GHz")
print(f"HAPS Velocity: {v} m/s")
print(f"Maximum Doppler Shift (f_D): {f_D:.2f} Hz")
print("-" * 40)

# 3. Calculate J_0(2 * pi * f_D * T_s) for each periodicity
for t_ms in periodicities_ms:
    # Convert milliseconds to seconds for the formula
    T_s = t_ms / 1000.0  
    
    # Calculate the time correlation using the Bessel function
    j0_val = sp.j0(2 * np.pi * f_D * T_s)
    
    print(f"For Ts = {t_ms:2d} ms ({T_s:.3f} s):")
    print(f"  Channel Correlation (J_0) = {j0_val:.4f}\n")