%% Generalized_Eig_Decomp.m
clear; clc;

% --- 1. Configuration (Match your saved file) ---
targetFreq_GHz = 2.0625;  % Note: '%.0f' in your save command rounded 2.0625 to 2
asf = 2;             % Antenna spacing factor
n = 7;
grid_size = [n n]; % [Rows, Columns] 

% --- 2. Load the Data ---
filename = sprintf('Z_results/%dx%d_UPA_%.0fGHz_spacing%d_Z.mat', ...
    grid_size(1), grid_size(2), targetFreq_GHz, asf);

if ~isfile(filename)
    error('File not found: %s', filename);
end

fprintf('Loading Z-Matrix from: %s\n', filename);
data = load(filename);
Z_T = data.Z_matrix;

% --- 3. Separate Real (R_T) and Imaginary (X_T) Parts ---
% Z_T = R_T + j * X_T
R_T = real(Z_T);
X_T = imag(Z_T);

fprintf('\n------------------------------------------------\n');
fprintf('Matrices Extracted:\n');
fprintf('R_T (Resistance): %dx%d\n', size(R_T));
fprintf('X_T (Reactance):  %dx%d\n', size(X_T));

% --- 4. Compute Generalized Eigendecomposition ---
% Equation: R_T * U_T = X_T * U_T * Lambda_T
% In MATLAB: [V, D] = eig(A, B) solves A*V = B*V*D
% Here: A = R_T, B = X_T

disp('Computing Generalized Eigenvalues [eig(R_T, X_T)]...');
[U_T, Lambda_T_Matrix] = eig(R_T, X_T);

% Extract diagonal eigenvalues for easier viewing
lambda_values = diag(Lambda_T_Matrix);
[lambda_sorted, sort_idx] = sort(lambda_values, 'descend');
% 3. CRITICAL: Reorder the Eigenvectors (U_T) using the same index
U_T_sorted = U_T(:, sort_idx);

% --- 5. Display Results ---
fprintf('\nGeneralized Eigenvalues (Lambda_T):\n');
disp(lambda_sorted);

% fprintf('Eigenvectors (Columns of U_T):\n');
% disp(U_T);

% 1. Create the folder if it doesn't exist
output_folder = 'eigen_result';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    fprintf('Folder created: %s\n', output_folder);
end

% 3. Construct Filename
filename_str = sprintf('%dx%d_UPA_%.0fGHz_spacing%d_eigen.mat', ...
    grid_size(1), grid_size(2), targetFreq_GHz, asf);
full_path = fullfile(output_folder, filename_str);

% 4. Save variables
% We save the sorted versions as the primary 'U_T' and 'Lambda_T' for ease of use later
save(full_path, 'U_T_sorted', 'lambda_sorted', 'grid_size', 'targetFreq_GHz', 'asf');

fprintf('Successfully saved sorted eigen-data to:\n%s\n', full_path);