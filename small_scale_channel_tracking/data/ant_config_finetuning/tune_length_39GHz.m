%% tune_length_39GHz.m
clear;

% 1. Define Target and Sweep Range
targetFreq = 38.75e9;
f_sweep = linspace(38.5e9, 39.2e9, 71); % Zoomed in!

% 2. Lock in our perfect Feed Offset
best_feed = 0.00090; % 0.90 mm

% 3. Test slightly longer lengths to hit exactly 38.75 GHz
lengths_to_test = [0.002375, 0.002377, 0.002379]; 

fig = figure('Visible', 'off');
hold on; grid on;
colors = {'r', 'b', 'm'};

fprintf('\n=== Final Length Micro-Tuning Sweep ===\n');

for i = 1:length(lengths_to_test)
    L = lengths_to_test(i);
    fprintf('Testing Length = %.4f mm...\n', L*1000);
    
    p = patchMicrostrip;
    p.Length = L;
    p.Width = 0.0031;
    p.Height = 0.0005;
    
    p.Substrate = dielectric('Name', 'Custom_Substrate');
    p.Substrate.EpsilonR = 2.2;
    p.Substrate.Thickness = 0.0005;
    
    p.GroundPlaneLength = 0.005;
    p.GroundPlaneWidth = 0.0058;
    p.FeedOffset = [best_feed, 0]; % Using the golden 0.90mm offset

    c = 299792458;
    mesh(p, 'MaxEdgeLength', (c/targetFreq) / 6);
    
    S_obj = sparameters(p, f_sweep);
    S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
    
    [min_S11, min_idx] = min(S11_dB);
    fprintf('  -> Peak at %.4f GHz (S11 = %.2f dB)\n', f_sweep(min_idx)/1e9, min_S11);
    
    plot(f_sweep/1e9, S11_dB, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('L = %.4f mm', L*1000));
end

xline(targetFreq/1e9, 'g--', 'Target 38.75 GHz', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-10, 'k:', 'Good Match (-10 dB)', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
legend('Location', 'best');
title('Final Patch Length Micro-Tuning (Offset = 0.90 mm)');

output_dir = 'S_results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
filename = sprintf('%s/Final_Length_Tuning_38.75GHz.png', output_dir);
print(fig, filename, '-dpng', '-r300');
close(fig);

fprintf('\nOptimization plot successfully saved to: %s\n', filename);