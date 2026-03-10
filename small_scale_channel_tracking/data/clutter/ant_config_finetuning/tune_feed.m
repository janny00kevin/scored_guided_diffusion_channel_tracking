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

fprintf('\n=== Feed Offset Optimization Sweep for %.4f GHz ===\n', targetFreq_GHz);

% 2. Dynamic Setup based on Frequency
if targetFreq_GHz == 38.75
    f_sweep = linspace(38.6e9, 38.9e9, 61);
    best_L = 0.002360; % Best length for 39 GHz
    offsets_to_test = [0.00085, 0.00090, 0.00095, 0.00100]; 
    
    W = 0.0031;
    H = 0.0005;
    EpsR = 2.2;
    Sub_name = 'Custom_Substrate';
    GP_L = 0.005;
    GP_W = 0.0058;
    mesh_len = (299792458/targetFreq) / 10;
    
elseif targetFreq_GHz == 2.0625
    % Keep the narrowed sweep
    f_sweep = linspace(2.04e9, 2.09e9, 51);
    
    best_L = 0.03867; % Your selected best length!
    
    % Final micro-tuning the feed offset around 10.3 mm
    offsets_to_test = [0.0102, 0.0103, 0.0104]; 
    
    W = 0.0442;
    H = 0.0014;
    EpsR = 4.4;
    Sub_name = 'FR4';
    
    % 1 lambda ground plane
    GP_L = 0.1455; 
    GP_W = 0.1455;
    
    mesh_len = 0.012;
else
    error('Unsupported frequency.');
end

S11_results = zeros(length(offsets_to_test), length(f_sweep));
fig = figure('Visible', 'off');
hold on; grid on;
colors = {'r', 'b', 'g', 'm', 'c'};

for i = 1:length(offsets_to_test)
    feed_x = offsets_to_test(i);
    fprintf('Testing Feed Offset = %.1f mm...\n', feed_x*1000);
    
    p = patchMicrostrip;
    p.Length = best_L;
    p.Width = W;
    p.Height = H;
    
    if strcmp(Sub_name, 'FR4')
        p.Substrate = dielectric('FR4');
    else
        p.Substrate = dielectric('Name', Sub_name);
        p.Substrate.Thickness = H;
    end
    p.Substrate.EpsilonR = EpsR;
    
    p.GroundPlaneLength = GP_L;
    p.GroundPlaneWidth = GP_W;
    p.FeedOffset = [feed_x, 0]; 

    c = 299792458;
    mesh(p, 'MaxEdgeLength', (c/targetFreq) / 6);
    
    S_obj = sparameters(p, f_sweep);
    S11_dB = squeeze(20*log10(abs(S_obj.Parameters(1,1,:))));
    S11_results(i, :) = S11_dB;
    
    [min_S11, min_idx] = min(S11_dB);
    Z_target = impedance(p, targetFreq);
    
    fprintf('  -> Peak S11 = %.2f dB at %.4f GHz | Z = %.2f %+.2fi Ohms\n', ...
        min_S11, f_sweep(min_idx)/1e9, real(Z_target), imag(Z_target));
        
    plot(f_sweep/1e9, S11_dB, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('Offset = %.1f mm', feed_x*1000));
end
