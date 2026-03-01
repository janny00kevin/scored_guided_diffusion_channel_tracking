%% Generalized_Eig_Decomp_Sweep.m
clear;

% ==========================================
% 1. Configuration
% ==========================================
freqs_GHz = [38.75];
% freqs_GHz = [2.0625];
grid_size = [3, 3];  % Currently testing 2x2
asf = 2;             % Antenna spacing factor

% Input/Output Directories
input_dir = 'Z_results';
output_dir = 'eigen_result';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('--- Starting GEVD Sweep for %d Frequencies ---\n', length(freqs_GHz));

% ==========================================
% 2. Main Loop
% ==========================================
for i = 1:length(freqs_GHz)
    targetFreq = freqs_GHz(i);

    % --- A. Load Z-Matrix ---
    % Filename format: "2x2_UPA_38.65GHz_Z.mat"
    filename = sprintf('%dx%d_UPA_%.2fGHz_Z.mat', ...
        grid_size(1), grid_size(2), targetFreq);
    full_path = fullfile(input_dir, filename);

    if ~isfile(full_path)
        warning('File not found: %s. Skipping...', filename);
        continue;
    end

    fprintf('[%d/%d] Processing %.2f GHz...', i, length(freqs_GHz), targetFreq);
    data = load(full_path);
    Z_T = data.Z_matrix;
    % --- B. Separate R and X ---
    R_T = real(Z_T);
    X_T = imag(Z_T);

    % --- C. Solve Generalized Eigenvalue Problem ---
    % PDF Eq (4): X_T * u = lambda * R_T * u
    % This defines lambda = X/R (Characteristic Values)
    % Ideally, lambda close to 0 is resonant.
    [U_raw, Lambda_Matrix] = eig(X_T, R_T);

    lambda_values = diag(Lambda_Matrix);

    % --- D. Normalization (PDF Eq 5) ---
    % Requirement: ||u_k||_2 = 1 (Euclidean Norm)
    [N, M] = size(U_raw);
    U_norm = zeros(N, M);

    for k = 1:M
        u_k = U_raw(:, k);
        % Divide the vector by its L2 norm to make its length exactly 1
        U_norm(:, k) = u_k / norm(u_k);
    end
    % --- E. Sorting (Initial) ---
    % We sort by Magnitude of lambda (Smallest |X/R| first)
    % This puts "Resonant" (or ghost resonant) modes at index 1
    % And "High Q/Capacitive" modes at the end.
    [~, sort_idx] = sort(abs(lambda_values), 'ascend');

    lambda_sorted = lambda_values(sort_idx);
    U_T_sorted = U_norm(:, sort_idx);

    % --- F. Save Results ---
    % Format: "2x2_UPA_38.65GHz_spacing2_eigen.mat"
    out_name = sprintf('%dx%d_UPA_%.2fGHz_eigen.mat', ...
        grid_size(1), grid_size(2), targetFreq);
    save_path = fullfile(output_dir, out_name);

    save(save_path, 'U_T_sorted', 'lambda_sorted', 'freqs_GHz', 'grid_size');
    fprintf(' Done. Saved to %s\n', out_name);
end

fprintf('--- Sweep Complete ---\n');
