%% tune_feed_39GHz.m
clear;

% 1. Target and Sweep Range (Slightly wider to catch upward shifts)
targetFreq = 38.75e9;
f_sweep = linspace(38.0e9, 40.0e9, 41); 

% 2. Lock in our optimized length
best_L = 0.002243; % 2.243 mm

% 3. Follow the trend OUTWARD
% Testing 0.55, 0.65, 0.75, 0.85, 0.95 mm
offsets_to_test = [0.00055, 0.00065, 0.00075, 0.00085, 0.00095]; 

% Pre-allocate storage
S11_results = zeros(length(offsets_to_test), length(f_sweep));

fprintf('\n=== Starting Feed Offset Optimization Sweep ===\n');

for i = 1:length(offsets_to_test)
    feed_x = offsets_to_test(i);
    fprintf('Testing Feed Offset = %.2f mm...\n', feed_x*1000);
    
    % Build Antenna
    p = patchMicrostrip;
    p.Length = best_L;
    p.Width = 0.0031;
    p.Height = 0.0005;
    
    p.Substrate = dielectric('Name', 'Custom_Substrate');
    p.Substrate.EpsilonR = 2.2;
    p.Substrate.Thickness = 0.0005;
    
    p.GroundPlaneLength = 0.005;
    p.GroundPlaneWidth = 0.0058;
    p.FeedOffset = [feed_x, 0]; % Applying the changing offset

    % Fast mesh setting
    c = 299792458;
    mesh(p, 'MaxEdgeLength', (c/targetFreq) / 6);
    
    % Calculate S-parameters
    S_obj = sparameters(p, f_sweep);
    S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
    
    % Store for plotting later
    S11_results(i, :) = S11_dB;
    
    % Print peak to console
    [min_S11, min_idx] = min(S11_dB);
    fprintf('  -> Peak S11 = %.2f dB at %.3f GHz\n', min_S11, f_sweep(min_idx)/1e9);
end

% --- Plot everything ---
fig = figure('Visible', 'off');
hold on; grid on;
colors = {'r', 'b', 'g', 'm', 'c'};

for i = 1:length(offsets_to_test)
    plot(f_sweep/1e9, S11_results(i, :), 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('Offset = %.2f mm', offsets_to_test(i)*1000));
end

xline(targetFreq/1e9, 'k--', 'Target 38.75 GHz', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-10, 'k:', 'Good Match (-10 dB)', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
legend('Location', 'best');
title('Feed Offset Optimization for 38.75 GHz');

% Save the plot
output_dir = 'S_results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
filename = sprintf('%s/Feed_Tuning_38.75GHz.png', output_dir);
print(fig, filename, '-dpng', '-r300');
close(fig);

fprintf('\nFeed Optimization plot saved to: %s\n', filename);