%% ULA_patch_Sweep.m
clear; clc;

% --- Configuration ---
asf = 2;               % lambda/2 spacing
num_elements = 2;
targetFreq = 2.0625e9;

% 1. Create Element & Array
p_element = createPatchAntenna_2GHz();
c = 299792458;
lambda = c / targetFreq;

ula = linearArray;
ula.Element = p_element;
ula.NumElements = num_elements;
ula.ElementSpacing = lambda / asf;

% 2. Define Frequency Range for the Sweep
% We scan +/- 100 MHz around the target to find the new peak
freq_scan = linspace(1.95e9, 2.3e9, 21);

% 3. Run Simulation (Sweep)
disp('Sweeping frequency to find new resonance...');
mesh(ula, 'MaxEdgeLength', lambda/10);
S_obj = sparameters(ula, freq_scan);

% 4. Plot S11 (Reflection Coefficient)
h_fig = figure;
% 2. Plotting (Same as before)
s11_data = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
plot(freq_scan/1e9, s11_data, 'b-', 'LineWidth', 2);
grid on; hold on;
xline(targetFreq/1e9, 'r--', 'Original Target (2.06 GHz)');
yline(-10, 'k--', '-10dB Threshold');
xlabel('Frequency (GHz)');
ylabel('Magnitude (dB)');
title(['Array S11: Check for Frequency Shift (N=' num2str(num_elements) ')']);

% 3. Save and Close using the handle
saveas(h_fig, 'plot_S/Array_S11_sweep.png');
close(h_fig);

% 5. Find the ACTUAL resonant frequency
[min_val, min_idx] = min(s11_data);
new_resonance = freq_scan(min_idx);
fprintf('\n------------------------------------------------\n');
fprintf('Original Target: %.4f GHz\n', targetFreq/1e9);
fprintf('New Resonance:   %.4f GHz\n', new_resonance/1e9);
fprintf('Shift Amount:    %.2f MHz\n', (new_resonance - targetFreq)/1e6);
fprintf('------------------------------------------------\n');

% 6. Calculate Z at the NEW resonance vs OLD target
Z_target = impedance(ula, targetFreq);
Z_peak   = impedance(ula, new_resonance);

disp('Z11 at Original Target (2.06 GHz):');
disp(Z_target(1,1));
disp('Z11 at New Peak (Actual Resonance):');
disp(Z_peak(1,1));