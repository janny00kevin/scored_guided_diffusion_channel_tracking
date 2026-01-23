%% Plot_MultiFreq_Energy_Comparison.m
clear;

% --- 1. Configurations ---
freqs = [2, 28, 39];         % Frequencies in GHz
grid_size = [7 7];           % 7x7 UPA
asf = 2;                     % Spacing factor
input_folder = 'eigen_result';
colors = {'b', 'r', 'g'};    % Colors for 2, 28, 39 GHz
markers_80 = 'o';            % Circle for 80%
markers_90 = 'p';            % Pentagram for 90%

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

% --- 2. Plotting Function Definition ---
% We define a helper logic to plot so we can reuse it for both figures
plot_type = {'Original Order', 'Magnitude Sorted Order'};

for fig_idx = 1:2
    % Initialize Figure (Visible off to run silently, set 'on' to see)
    fig = figure('Visible', 'off');
    hold on; grid on;
    
    fprintf('\nGenerating Figure %d: %s...\n', fig_idx, plot_type{fig_idx});
    
    for i = 1:length(freqs)
        lambda_raw = data_storage{i};
        if isempty(lambda_raw), continue; end
        
        % --- PROCESSING LOGIC ---
        if fig_idx == 1
            % CASE 1: Original Order (as saved in file)
            % The file was sorted by signed value (10... -10)
            % We just take the magnitude of that sequence
            lambda_seq = abs(lambda_raw);
        else
            % CASE 2: Absolute Order (Re-sorted by Magnitude)
            % We sort the raw lambdas by their absolute magnitude first
            [~, sort_idx] = sort(abs(lambda_raw), 'descend');
            lambda_seq = abs(lambda_raw(sort_idx));
        end
        
        % Calculate Cumulative Energy
        cum_energy = cumsum(lambda_seq) / sum(lambda_seq);
        num_modes = 1:length(cum_energy);
        
        % Plot Curve
        plot(num_modes, cum_energy, 'Color', colors{i}, 'LineWidth', 2, ...
            'DisplayName', [num2str(freqs(i)) ' GHz']);
        
        % --- ADD POINTS (80% & 90%) ---
        idx_80 = find(cum_energy >= 0.80, 1);
        idx_90 = find(cum_energy >= 0.90, 1);
        
        % % Plot 80% Point
        % plot(idx_80, cum_energy(idx_80), markers_80, ...
        %     'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k', ...
        %     'MarkerSize', 8, 'HandleVisibility', 'off');
        
        % % Plot 90% Point
        % plot(idx_90, cum_energy(idx_90), markers_90, ...
        %     'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k', ...
        %     'MarkerSize', 12, 'HandleVisibility', 'off');
            
        fprintf('  %d GHz: 80%% @ r=%d, 90%% @ r=%d\n', freqs(i), idx_80, idx_90);
    end
    
    % --- Formatting ---
    yline(0.90, 'k--', '90% Threshold', 'LabelHorizontalAlignment', 'left', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    yline(0.80, 'k:', '80% Threshold', 'LabelHorizontalAlignment', 'left', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    xlabel('Number of Eigenvalues');
    ylabel('Cumulative Energy Ratio');
    title(['Cumulative Energy: ' plot_type{fig_idx}]);
    legend('Location', 'southeast');
    xlim([1, 49]); % Fixed for 7x7 array
    ylim([0, 1.05]);
    
    % --- Save Figure ---
    % Create filename based on type
    fname_suffix = strrep(plot_type{fig_idx}, ' ', '_');
    save_path = fullfile(input_folder, ['Energy_Comparison_' fname_suffix '.png']);
    saveas(fig, save_path);
    close(fig);
    fprintf('Saved: %s\n', save_path);
end

fprintf('\nDone! Check the "%s" folder for two images.\n', input_folder);