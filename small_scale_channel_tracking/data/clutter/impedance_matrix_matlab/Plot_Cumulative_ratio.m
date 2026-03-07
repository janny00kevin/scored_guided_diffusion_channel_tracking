%% Plot_Cumulative_ratio.m
clear;

% --- 1. Configurations ---
freqs = [2, 28, 39];         % Frequencies in GHz
grid_size = [7 7];           % 7x7 UPA
asf = 2;                     % Spacing factor
input_folder = 'eigen_result';
colors = {'b', 'r', 'g'};    % Colors for 2, 28, 39 GHz
marker_style = 'o';          % Circle for ALL points now

% Prepare storage for data to avoid reloading files twice
data_storage = cell(length(freqs), 1);

% Load Data First
fprintf('Loading data files...\n');
for i = 1:length(freqs)
    f = freqs(i);
    filename = sprintf('%dx%d_UPA_%.0fGHz_spacing%d_eigen.mat', ...
        grid_size(1), grid_size(2), f, asf);
    full_path = fullfile(input_folder, filename);
    
    if isfile(full_path)
        loaded_struct = load(full_path);
        data_storage{i} = loaded_struct.lambda_sorted;
    else
        warning('File not found: %s', filename);
        data_storage{i} = [];
    end
end

% --- 2. Plotting ---
% Removed specific 'Position' argument as requested
fig = figure('Visible', 'off'); 
hold on; grid on;

fprintf('\nGenerating Labeled Cumulative Energy Plot...\n');

for i = 1:length(freqs)
    lambda_raw = data_storage{i};
    if isempty(lambda_raw), continue; end
    
    % --- PROCESSING LOGIC ---
    % Take absolute value (energy is scalar magnitude)
    lambda_seq = abs(lambda_raw);
    
    % Calculate Cumulative Energy
    cum_energy = cumsum(lambda_seq) / sum(lambda_seq);
    num_modes = 1:length(cum_energy);
    
    % Plot Main Curve
    plot(num_modes, cum_energy, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', [num2str(freqs(i)) ' GHz']);
    
    % --- FIND POINTS ---
    idx_80 = find(cum_energy >= 0.80, 1);
    idx_90 = find(cum_energy >= 0.90, 1);
    
    % --- PLOT MARKERS (All Circles) ---
    % 80% Point
    plot(idx_80, cum_energy(idx_80), marker_style, ...
        'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 7, 'HandleVisibility', 'off');
    
    % 90% Point
    plot(idx_90, cum_energy(idx_90), marker_style, ...
        'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 7, 'HandleVisibility', 'off');
        
    % --- ADD TEXT LABELS (e.g. "25th") ---
    % Label for 80%
    text(idx_80, cum_energy(idx_80), sprintf('  %dth', idx_80), ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'FontSize', 9, 'Color', 'k');
        
    % Label for 90%
    text(idx_90, cum_energy(idx_90), sprintf('  %dth', idx_90), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 9, 'Color', 'k');
            
    fprintf('  %d GHz: 80%% @ %dth, 90%% @ %dth\n', freqs(i), idx_80, idx_90);
end

% --- Formatting ---
yline(0.90, 'k--', '90%', 'LabelHorizontalAlignment', 'left', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(0.80, 'k:', '80%', 'LabelHorizontalAlignment', 'left', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Number of Eigenvalues');
ylabel('Cumulative Energy Ratio');
title('Cumulative Energy (Magnitude Sorted)');
legend('Location', 'southeast');
xlim([1, grid_size(1)*grid_size(2)]); % Adjusted for general grid size
ylim([0, 1.05]);

% --- Save Figure ---
file_name = sprintf('%dx%d_Energy_Comparison_Magnitude_Sorted.png', grid_size(1), grid_size(2));
save_path = fullfile(input_folder, file_name);
saveas(fig, save_path);
close(fig);
fprintf('Saved: %s\n', save_path);