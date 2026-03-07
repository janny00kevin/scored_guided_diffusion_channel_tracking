%% Generalized_Eig_Decomp.m
clear; clc;

% --- Configuration ---
freqs_GHz = [2.065, 38.75];
grid_size = [7, 7];

if ~exist('eigen_result', 'dir'), mkdir('eigen_result'); end

for i = 1:length(freqs_GHz)
    targetFreq = freqs_GHz(i);
    filename = sprintf('Z_results/%dx%d_UPA_%.3fGHz_Z.mat', grid_size(1), grid_size(2), targetFreq);
    
    if ~isfile(filename)
        warning('File %s not found. Skipping...', filename);
        continue;
    end
    
    fprintf('Processing GEVD for %.3f GHz...\n', targetFreq);
    data = load(filename);
    Z_T = data.Z_matrix;
    
    R_T = real(Z_T);
    X_T = imag(Z_T);
    
    % 1. GEVD formulation: X*U = R*U*Lambda
    [U_raw, Lambda_Matrix] = eig(X_T, R_T);
    lambda_values = diag(Lambda_Matrix);
    
    % 2. Normalize eigenvectors by their Euclidean norm
    [N, M] = size(U_raw);
    U_norm = zeros(N, M);
    for k = 1:M
        U_norm(:, k) = U_raw(:, k) / norm(U_raw(:, k));
    end
    
    % 3. Sort ascending (so the smallest eigenvalues come first)
    [~, sort_idx] = sort(abs(lambda_values), 'ascend');
    lambda_sorted = lambda_values(sort_idx);
    U_T_sorted = U_norm(:, sort_idx);
    
    % 4. Save
    out_name = sprintf('eigen_result/%dx%d_UPA_%.3fGHz_eigen.mat', grid_size(1), grid_size(2), targetFreq);
    save(out_name, 'U_T_sorted', 'lambda_sorted', 'targetFreq', 'grid_size');
    fprintf('Saved sorted eigen-data to %s\n', out_name);
end

fprintf('All frequencies completed.\n');