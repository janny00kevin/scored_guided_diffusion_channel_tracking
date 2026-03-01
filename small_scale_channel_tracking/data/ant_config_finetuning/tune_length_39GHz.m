%% tune_length_39GHz.m
clear; 

% 1. Define Target and Sweep Range
targetFreq = 38.75e9;
f_sweep = linspace(37.5e9, 40e9, 51);

% 2. Define Lengths to Test 
% Testing 2.23 mm, 2.25 mm, and 2.27 mm
lengths_to_test = [0.00223, 0.00225, 0.00227]; 

% Prepare the plot (invisible figure)
fig = figure('Visible', 'off');
hold on; grid on;
colors = {'r', 'b', 'm'};

fprintf('\n=== Starting Length Optimization Sweep ===\n');

for i = 1:length(lengths_to_test)
    L = lengths_to_test(i);
    fprintf('Testing Length = %.3f mm...\n', L*1000);
    
    % Build Antenna
    p = patchMicrostrip;
    p.Length = L;
    p.Width = 0.0031;
    p.Height = 0.0005;
    
    p.Substrate = dielectric('Name', 'Custom_Substrate');
    p.Substrate.EpsilonR = 2.2;
    p.Substrate.Thickness = 0.0005;
    
    p.GroundPlaneLength = 0.005;
    p.GroundPlaneWidth = 0.0058;
    p.FeedOffset = [0.00045, 0]; % Keeping feed offset locked for now

    % Mesh and calculate
    c = 299792458;
    mesh(p, 'MaxEdgeLength', (c/targetFreq) / 10);
    
    S_obj = sparameters(p, f_sweep);
    S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
    
    % Find peak for console output
    [min_S11, min_idx] = min(S11_dB);
    fprintf('  -> Peak at %.3f GHz (S11 = %.2f dB)\n', f_sweep(min_idx)/1e9, min_S11);
    
    % Add to plot
    plot(f_sweep/1e9, S11_dB, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('L = %.3f mm', L*1000));
end

% Formatting the Plot
xline(targetFreq/1e9, 'g--', 'Target 38.75 GHz', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-10, 'k:', 'Good Match (-10 dB)', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
legend('Location', 'best');
title('Patch Length Optimization for 38.75 GHz');

% Save the plot
output_dir = 'S_results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
filename = sprintf('%s/Length_Tuning_38.75GHz.png', output_dir);
print(fig, filename, '-dpng', '-r300');
close(fig);

fprintf('\nOptimization plot successfully saved to: %s\n', filename);
fprintf('Check the image to see which length aligns best with the green line!\n');