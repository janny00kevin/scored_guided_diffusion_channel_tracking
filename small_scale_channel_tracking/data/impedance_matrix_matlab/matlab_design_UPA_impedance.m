% =========================================================================
%  Automated Patch Array (UPA) Analysis Script
%  Task: Design UPA, Compute Z, Compute GEVD, Save Results
% =========================================================================

% 1. Setup Parameters
frequencies = [2.0625e9, 28e9, 38.75e9]; % The 3 cases
grid_size = [7 7];                       % 2x2 Array

% 2. Create output directories if they don't exist
if ~exist('Z_results', 'dir')
    mkdir('Z_results');
end
if ~exist('eigen_result', 'dir')
    mkdir('eigen_result');
end

% 3. Main Loop
for f_idx = 1:length(frequencies)
    targetFreq = frequencies(f_idx);
    targetFreq_GHz = targetFreq / 1e9;
    
    fprintf('------------------------------------------------\n');
    fprintf('Processing Case %d: %.4f GHz\n', f_idx, targetFreq_GHz);
    
    % --- Step 1: Generate the UPA Object ---
    % Explicitly define the element as a Patch
    elementObj = patchMicrostrip;
    
    % Create the container
    ra = rectangularArray('Element', elementObj, 'Size', grid_size);
    
    % Design the array (tunes patch dims and spacing automatically)
    % passing elementObj again ensures it stays a patch
    upa = design(ra, targetFreq, elementObj); 

    % Define a reasonable mesh size
    c = 299792458;
    lambda = c / targetFreq;
    meshSize = lambda / 8;
    mesh(upa, 'MaxEdgeLength', meshSize);
    
    % --- Step 2: Compute Impedance (Z Matrix) ---
    fprintf('  > Computing Impedance Matrix (Z)...\n');
    tic;
    % 1. Calculate S-parameters (Returns the full N x N matrix)
    S_obj = sparameters(upa, targetFreq);
    % 2. Convert S-parameters to Z-parameters
    Z_matrix = s2z(S_obj.Parameters, S_obj.Impedance);
    elapsedTime = toc;
    fprintf('    (Calculation took %.2f seconds)\n', elapsedTime);
    
    % Save Z Matrix
    % Changed %.0f to %.4g so 2.0625 doesn't get rounded to 2
    filename_Z = sprintf('Z_results/%dx%d_UPA_%.0fGHz_Z_matDesign.mat', ...
        grid_size(1), grid_size(2), targetFreq_GHz);
        
    save(filename_Z, 'Z_matrix', 'targetFreq');
    fprintf('  > Saved Z to: %s\n', filename_Z);
    
    % --- Step 3: Compute GEVD (Generalized Eigenvalue Decomposition) ---
    fprintf('  > Computing GEVD of Z = R + jX...\n');
    
    R_T = real(Z_matrix);
    X_T = imag(Z_matrix);
    
    % Solving Generalized Eigenvalue Problem: R*v = lambda*X*v
    [U_T, Lambda_T_Matrix] = eig(R_T, X_T);

    % Extract diagonal eigenvalues
    lambda_values = diag(Lambda_T_Matrix);

    % Sort by MAGNITUDE (descending)
    [~, sort_idx] = sort(abs(lambda_values), 'descend');
    lambda_sorted = lambda_values(sort_idx);

    % Reorder Eigenvectors to match
    U_T_sorted = U_T(:, sort_idx);

    % Normalize each eigenvector to have unit norm
    for i = 1:size(U_T_sorted, 2)
        U_T_sorted(:, i) = U_T_sorted(:, i) / norm(U_T_sorted(:, i));
    end
    
    % Save Eigen Results
    filename_eig = sprintf('eigen_result/%dx%d_UPA_%.0fGHz_eigen_matDesign.mat', ...
        grid_size(1), grid_size(2), targetFreq_GHz);
        
    save(filename_eig, 'U_T_sorted', 'lambda_sorted', 'targetFreq_GHz');
    fprintf('  > Saved GEVD to: %s\n', filename_eig);
end

fprintf('------------------------------------------------\n');
fprintf('All tasks completed successfully.\n');