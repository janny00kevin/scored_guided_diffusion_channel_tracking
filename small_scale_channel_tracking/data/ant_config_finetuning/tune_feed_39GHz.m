%% tune_feed_39GHz.m
clear;

% 1. High-Resolution Sweep Range
targetFreq = 38.75e9;
f_sweep = linspace(38.6e9, 38.9e9, 61); % Zoomed in!

% 2. Lock in our perfect length!
best_L = 0.002360; % 2.360 mm

% 3. Test larger feed offsets to bring 68 Ohms down to 50 Ohms
offsets_to_test = [0.00085, 0.00090, 0.00095, 0.00100]; 

S11_results = zeros(length(offsets_to_test), length(f_sweep));

fprintf('\n=== Final Feed Offset Micro-Tuning ===\n');

for i = 1:length(offsets_to_test)
    feed_x = offsets_to_test(i);
    fprintf('Testing Feed Offset = %.2f mm...\n', feed_x*1000);
    
    p = patchMicrostrip;
    p.Length = best_L;
    p.Width = 0.0031;
    p.Height = 0.0005;
    
    p.Substrate = dielectric('Name', 'Custom_Substrate');
    p.Substrate.EpsilonR = 2.2;
    p.Substrate.Thickness = 0.0005;
    
    p.GroundPlaneLength = 0.005;
    p.GroundPlaneWidth = 0.0058;
    p.FeedOffset = [feed_x, 0]; 

    c = 299792458;
    mesh(p, 'MaxEdgeLength', (c/targetFreq) / 6);
    
    S_obj = sparameters(p, f_sweep);
    S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
    S11_results(i, :) = S11_dB;
    
    [min_S11, min_idx] = min(S11_dB);
    Z_target = impedance(p, targetFreq);
    
    % Print both the peak and the impedance at our target!
    fprintf('  -> Peak S11 = %.2f dB at %.3f GHz | Z = %.2f %+.2fi Ohms\n', ...
        min_S11, f_sweep(min_idx)/1e9, real(Z_target), imag(Z_target));
end

fig = figure('Visible', 'off');
hold on; grid on;
colors = {'r', 'b', 'g', 'm'};

for i = 1:length(offsets_to_test)
    plot(f_sweep/1e9, S11_results(i, :), 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('Offset = %.2f mm', offsets_to_test(i)*1000));
end

xline(targetFreq/1e9, 'k--', 'Target 38.75 GHz', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-10, 'k:', 'Good Match (-10 dB)', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-20, 'k:', 'Excellent Match (-20 dB)', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
legend('Location', 'best');
title('Final Feed Offset Micro-Tuning');

output_dir = 'S_results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
filename = sprintf('%s/Final_Feed_Tuning_38.75GHz.png', output_dir);
print(fig, filename, '-dpng', '-r300');
close(fig);

fprintf('\nFeed Optimization plot saved to: %s\n', filename);