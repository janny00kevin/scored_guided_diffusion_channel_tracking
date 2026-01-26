%% Generalized_Eig_Decomp_Reconstruction.m
clear; clc;

% --- 1. Configuration ---
targetFreq_GHz = 28; % Target Frequency
asf = 2;             % Antenna spacing factor
n = 7;               % Array size (7x7)
grid_size = [n n];   % [Rows, Columns] 
Z0 = 50;             % Characteristic Impedance

% --- 2. Load the Data ---
filename = sprintf('Z_results/%dx%d_UPA_%.0fGHz_spacing%d_Z.mat', ...
    grid_size(1), grid_size(2), targetFreq_GHz, asf);

if ~isfile(filename)
    error('File not found: %s', filename);
end

fprintf('Loading Z-Matrix from: %s\n', filename);
data = load(filename);
Z_T = data.Z_matrix;
[NT, ~] = size(Z_T); % Total number of elements (49)

% --- 3. Compute "True" Coupling Matrix C_T (Reference) ---
% Formula: C_T = 1/2 * Re{ (Z_T + Z0*I)^-1 * Z_T * (Z_T + Z0*I)^-H }
I_NT = eye(NT);
A_term = Z_T + (Z0 * I_NT);
% Efficient calculation using backslash (\) and slash (/) instead of inv()
Inner_Product = (A_term \ Z_T) / (A_term'); 
C_T_True = 0.5 * real(Inner_Product);

fprintf('True Coupling Matrix C_T computed. Norm: %.4f\n', norm(C_T_True, 'fro'));

% --- 4. Compute Generalized Eigendecomposition ---
R_T = real(Z_T);
X_T = imag(Z_T);

disp('Computing Generalized Eigenvalues [eig(R_T, X_T)]...');
[U_T, Lambda_T_Matrix] = eig(R_T, X_T);

% Extract and Sort Eigenvalues/Vectors (Descending)
lambda_values = diag(Lambda_T_Matrix);
[lambda_sorted, sort_idx] = sort(lambda_values, 'descend');
U_T_sorted = U_T(:, sort_idx);

% --- 5. Reconstruction Loop (k = 2 to 40) ---
k_range = 2:40;
error_list = zeros(length(k_range), 1);

fprintf('\nStarting Reconstruction Loop (k = 2 to %d)...\n', max(k_range));

for i = 1:length(k_range)
    k = k_range(i);
    
    % A. Construct Reduced Matrices
    % Take first k columns of U
    U_k = U_T_sorted(:, 1:k); 
    % Take first k eigenvalues (diagonal matrix)
    Gamma_k = diag(lambda_sorted(1:k));
    
    % B. Reconstruct C_hat
    % Assumption: C_hat = U * Gamma * U^H
    C_hat = U_k * Gamma_k * U_k';
    
    % C. Compute Frobenius Norm Error
    % Error = || C_True - C_hat ||_F
    error_list(i) = norm(C_T_True - C_hat, 'fro');
end

% --- 6. Plotting ---
% Create results folder if needed
if ~exist('reconstruct_error', 'dir'), mkdir('reconstruct_error'); end

fig = figure('Visible', 'off'); 
plot(k_range, error_list, '-o', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
xlabel('Number of Eigenvectors Used ($r_T$)', 'Interpreter', 'latex');
ylabel('Reconstruction Error ($||\mathbf{C}_T - \widehat{\mathbf{C}}_T||_F$)', 'Interpreter', 'latex');
title({['Coupling Matrix Reconstruction Error (Frobenius Norm)']; ...
       [num2str(targetFreq_GHz) ' GHz, ' num2str(n) 'x' num2str(n) ' UPA']});

% % Add annotation for the minimum error found in this range
% [min_err, min_idx] = min(error_list);
% txt_str = sprintf('Min Error: %.4f at k=%d', min_err, k_range(min_idx));
% text(k_range(min_idx), min_err, txt_str, 'VerticalAlignment', 'bottom', 'BackgroundColor', 'w');

% Save the plot
plot_filename = sprintf('reconstruct_error/%dx%d_Reconstruction_Error_%.0fGHz.png', ...
    n, n, targetFreq_GHz);
saveas(fig, plot_filename);
fprintf('\nPlot saved to: %s\n', plot_filename);

% --- 7. Save Eigen Data (User Request) ---
output_folder = 'eigen_result';
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

eigen_filename = sprintf('%dx%d_UPA_%.0fGHz_spacing%d_eigen.mat', ...
    grid_size(1), grid_size(2), targetFreq_GHz, asf);
full_path_eigen = fullfile(output_folder, eigen_filename);

save(full_path_eigen, 'U_T_sorted', 'lambda_sorted', 'grid_size', 'targetFreq_GHz', 'asf', 'error_list', 'k_range');
fprintf('Eigen-data saved to: %s\n', full_path_eigen);