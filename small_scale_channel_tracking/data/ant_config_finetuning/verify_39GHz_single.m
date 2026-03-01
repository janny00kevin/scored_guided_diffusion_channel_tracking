%% verify_39GHz_single.m
clear;

% Add the path to your single_antenna folder
addpath('single_antenna');

% 1. Define target frequency and sweep range
targetFreq = 38.75e9; % 38.75 GHz
f_start = 38.0e9;
f_end = 39.5e9;
f_sweep = linspace(f_start, f_end, 61); 

% 2. Create the antenna object
p = createPatchAntenna_39GHz();

% Set mesh manually for consistent calculation
c = 299792458;
lambda = c / targetFreq;
mesh(p, 'MaxEdgeLength', lambda / 6);

% 3. Calculate S-parameters
fprintf('Calculating S-parameters around %.2f GHz...\n', targetFreq/1e9);
tic;
S_obj = sparameters(p, f_sweep);
toc;

% Extract S11 in dB
S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));

% 4. Plot the S11 Results (Invisible Figure)
fig = figure('Visible', 'off');
plot(f_sweep/1e9, S11_dB, 'b-', 'LineWidth', 2);
grid on; hold on;

% Add reference lines
xline(targetFreq/1e9, 'r--', ['Target (' num2str(targetFreq/1e9) ' GHz)'], 'LineWidth', 1.5);
yline(-10, 'k--', 'Good Matching Threshold (-10 dB)', 'LineWidth', 1.5);

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
title(sprintf('S11 Reflection Coefficient - %.2f GHz Patch Antenna', targetFreq/1e9));

% 5. Save the Figure
% Create directory if it doesn't exist
output_dir = 'S_results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Construct filename with important parameters
filename = sprintf('%s/SinglePatch_S11_Target%.2fGHz_Sweep%.1fto%.1fGHz.png', ...
                   output_dir, targetFreq/1e9, f_start/1e9, f_end/1e9);

% Save as high-resolution PNG
print(fig, filename, '-dpng', '-r300');
close(fig);
fprintf('\nPlot successfully saved to: %s\n', filename);

% 6. Find and report the actual resonant peak
[min_S11, min_idx] = min(S11_dB);
actual_resonance = f_sweep(min_idx);

fprintf('\n=== Verification Results ===\n');
fprintf('Target Frequency:        %.4f GHz\n', targetFreq/1e9);
fprintf('Actual Resonant Peak:    %.4f GHz\n', actual_resonance/1e9);
fprintf('S11 at Peak:             %.2f dB\n', min_S11);

% Check Z at target frequency
Z_target = impedance(p, targetFreq);
fprintf('Impedance at Target:     %.2f %+.2fi Ohms\n', real(Z_target), imag(Z_target));