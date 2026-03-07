%% verify_single_antenna.m
clear;

% Add the path to your single_antenna folder
addpath('single_antenna');

% 1. Configuration: Choose Frequency
% Set to 38 or 2
mode = 2;

if mode == 2
    targetFreq_GHz = 2.0625;
elseif mode == 38
    targetFreq_GHz = 38.75;
end
targetFreq = targetFreq_GHz * 1e9;

% 2. Dynamic Setup based on Frequency
if targetFreq_GHz == 38.75
    p = createPatchAntenna_39GHz();
    % Sweep range: +/- 0.75 GHz
    f_start = 38.0e9;
    f_end = 39.5e9;
elseif targetFreq_GHz == 2.0625
    p = createPatchAntenna_2GHz();
    % Sweep range: +/- 0.1 GHz
    f_start = 1.9625e9;
    f_end = 2.1625e9;
else
    error('Unsupported frequency. Please use 38.75 or 2.0625.');
end

% Set mesh manually for consistent calculation
c = 299792458;
lambda = c / targetFreq;
mesh(p, 'MaxEdgeLength', lambda / 6); % Finer mesh for refinement

% 3. Calculate S-parameters
fprintf('Calculating S-parameters around %.4f GHz...\n', targetFreq_GHz);
f_sweep = linspace(f_start, f_end, 61); 
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
xline(targetFreq_GHz, 'r--', ['Target (' num2str(targetFreq_GHz) ' GHz)'], 'LineWidth', 1.5);
yline(-10, 'k--', 'Matching Threshold (-10 dB)', 'LineWidth', 1.5);

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
title(sprintf('S11 Reflection Coefficient - %.4f GHz Patch Antenna', targetFreq_GHz));

% 5. Save the Figure
output_dir = 'S_results';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

filename = sprintf('%s/Verify_Single_%.2fGHz.png', output_dir, targetFreq_GHz);
print(fig, filename, '-dpng', '-r300');
close(fig);
fprintf('\nPlot successfully saved to: %s\n', filename);

% 6. Report the Results
[min_S11, min_idx] = min(S11_dB);
actual_resonance = f_sweep(min_idx);

fprintf('\n=== Verification Results ===\n');
fprintf('Target Frequency:        %.4f GHz\n', targetFreq_GHz);
fprintf('Actual Resonant Peak:    %.4f GHz\n', actual_resonance/1e9);
fprintf('S11 at Peak:             %.2f dB\n', min_S11);

% Check Z at target frequency
Z_target = impedance(p, targetFreq);
fprintf('Impedance at Target:     %.2f %+.2fi Ohms\n', real(Z_target), imag(Z_target));