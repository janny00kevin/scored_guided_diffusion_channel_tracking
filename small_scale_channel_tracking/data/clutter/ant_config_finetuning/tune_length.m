%% tune_length.m
clear;

% 1. Configuration: Choose Frequency
% Set to 38 or 2
mode = 2;

if mode == 2
    targetFreq_GHz = 2.0625;
elseif mode == 38
    targetFreq_GHz = 38.75;
end
targetFreq = targetFreq_GHz * 1e9;

fprintf('\n=== Length Optimization Sweep for %.4f GHz ===\n', targetFreq_GHz);

% 2. Dynamic Setup based on Frequency
if targetFreq_GHz == 38.75
    f_sweep = linspace(38.5e9, 39.2e9, 71);
    best_feed = 0.00090; % 0.90 mm
    lengths_to_test = [0.0357, 0.0358, 0.0359];
    
    W = 0.0031;
    H = 0.0005;
    EpsR = 2.2;
    Sub_name = 'Custom_Substrate';
    GP_L = 0.005;
    GP_W = 0.0058;
    
elseif targetFreq_GHz == 2.0625
    % Narrow frequency sweep around our target
    f_sweep = linspace(2.04e9, 2.09e9, 51);
    
    % Lock in your newly discovered perfect feed offset!
    best_feed = 0.0104; % 10.4 mm
    
    % Shorten the patch slightly to raise the frequency from 2.058 up to 2.0625
    lengths_to_test = [0.03855, 0.03860, 0.03865]; 
    
    W = 0.0442;
    H = 0.0014;
    EpsR = 4.4;
    Sub_name = 'FR4';
    
    % 1 lambda ground plane
    GP_L = 0.1455; 
    GP_W = 0.1455;
else
    error('Unsupported frequency.');
end

fig = figure('Visible', 'off');
hold on; grid on;
colors = {'r', 'b', 'm', 'g', 'c'};

for i = 1:length(lengths_to_test)
    L = lengths_to_test(i);
    fprintf('Testing Length = %.2f mm...\n', L*1000);
    
    p = patchMicrostrip;
    p.Length = L;
    p.Width = W;
    p.Height = H;
    
    % Assign proper substrate based on frequency
    if strcmp(Sub_name, 'FR4')
        p.Substrate = dielectric('FR4');
    else
        p.Substrate = dielectric('Name', Sub_name);
        p.Substrate.Thickness = H;
    end
    p.Substrate.EpsilonR = EpsR; 
    
    p.GroundPlaneLength = GP_L;
    p.GroundPlaneWidth = GP_W;
    p.FeedOffset = [best_feed, 0]; 

    c = 299792458;
    mesh(p, 'MaxEdgeLength', (c/targetFreq) / 6);
    
    S_obj = sparameters(p, f_sweep);
    S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
    
    [min_S11, min_idx] = min(S11_dB);
    fprintf('  -> Peak at %.4f GHz (S11 = %.2f dB)\n', f_sweep(min_idx)/1e9, min_S11);
    
    plot(f_sweep/1e9, S11_dB, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('L = %.2f mm', L*1000));
end

% Formatting the plot
xline(targetFreq_GHz, 'g--', sprintf('Target %.4f GHz', targetFreq_GHz), 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-10, 'k:', 'Good Match (-10 dB)', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Frequency (GHz)');
ylabel('S11 Magnitude (dB)');
legend('Location', 'best');
title(sprintf('Patch Length Optimization (Offset = %.2f mm)', best_feed*1000));

% Save output
output_dir = 'S_results';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end