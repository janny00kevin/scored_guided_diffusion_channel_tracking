clc; clear; close all;

%% --- 1. Path Setup ---
% Get the directory where this script is saved
script_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(script_dir, 'SE_vs_rT_result');

freq_ghz = 12;
mat_filename = sprintf('SE_vs_rT_%dGHz_data.mat', freq_ghz);
mat_filepath = fullfile(result_dir, mat_filename);

if ~exist(mat_filepath, 'file')
    error('Error: Could not find the data file at:\n%s\nRun your Python simulation first.', mat_filepath);
end

%% --- 2. Load Data ---
fprintf('Loading data from %s...\n', mat_filename);
data = load(mat_filepath);

% Extract core vectors
snr_list = data.SNR_LIST;
rt_range = data.RT_RANGE;

%% --- 3. Plotting Loop for Each SNR ---
for idx = 1:length(snr_list)
    snr = snr_list(idx);
    fprintf('\nGenerating editable figure for SNR = %d dB...\n', snr);
    
    % Reconstruct the dynamic naming string used by Python (e.g., 'n4', 'p0', 'p4')
    if snr < 0
        snr_str = sprintf('n%d', abs(snr));
    else
        snr_str = sprintf('p%d', snr);
    end
    
    % Dynamically fetch the variables from the loaded structure
    baseline = data.(sprintf('baseline_snr_%s', snr_str));
    res_ideal = data.(sprintf('ideal_snr_%s', snr_str));
    res_hybrid = data.(sprintf('hybrid_snr_%s', snr_str));
    
    %% --- 4. Generate the Figure (IEEE Clean Style by default) ---
    % Create a clean, standard aspect-ratio figure canvas
    h_fig = figure('Units', 'inches', 'Position', [2, 2, 5, 3.8]);
    hold on;
    
    % Plot curves with clean styling (thicker lines for easy viewing)
    plot(rt_range, res_ideal, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Modal Fully Digital');
    plot(rt_range, res_hybrid, 'g--o', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'g', 'DisplayName', 'Modal Hybrid w/ E');
    
    % Plot the Port full-digital ceiling limit
    yline(baseline, 'k:', 'LineWidth', 1.8, 'DisplayName', 'Port Bound');
    
    %% --- 5. Figure Details and Labeling ---
    xlabel('Number of Modes (r_T)', 'FontSize', 11, 'FontName', 'Helvetica');
    ylabel('Spectral Efficiency (bps/Hz)', 'FontSize', 11, 'FontName', 'Helvetica');
    title(sprintf('SE Analysis (SNR = %d dB, %d GHz Imperfect CSI)', snr, freq_ghz), 'FontSize', 11, 'FontWeight', 'bold');
    
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6);
    set(gca, 'FontSize', 10, 'FontName', 'Helvetica'); % Axis tick font sizes
    
    % Add clean legend without a bulky border box
    legend('Location', 'southeast', 'FontSize', 9, 'Box', 'off');
    
    hold off;
    
    %% --- 6. Save Simultaneously as Editable .fig and Publication-Quality .png ---
    % 1. Save the editable MATLAB layout file
    save_filename_fig = sprintf('SE_vs_rT_SNR_%sdB.fig', snr_str);
    save_filepath_fig = fullfile(result_dir, save_filename_fig);
    saveas(h_fig, save_filepath_fig);
    
    % 2. Save the raster image file (300 DPI minimum for IEEE print layout)
    save_filename_png = sprintf('SE_vs_rT_SNR_%sdB.png', snr_str);
    save_filepath_png = fullfile(result_dir, save_filename_png);
    exportgraphics(h_fig, save_filepath_png, 'Resolution', 300);
    
    % Close the active canvas to clear memory
    close(h_fig); 
    
    fprintf('Saved editable fig: %s\n', save_filename_fig);
    fprintf('Saved viewing png:  %s\n', save_filename_png);
end

fprintf('\nSuccess! All figures are safely saved and fully editable inside MATLAB.\n');