%% Generalized_Eig_Decomp.m
clear; clc;

% --- 1. Configuration (Match your saved file) ---
targetFreq_GHz = 39;  % Note: '%.0f' in your save command rounded 2.0625 to 2
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

% % --- 6. Calculate Successive Eigenvalue Ratios ---
% % Formula: ratio(i) = lambda(i) / lambda(i+1)
% num_ratios = length(lambda_sorted) - 1;
% lambda_ratios = zeros(num_ratios, 1);

% for i = 1:num_ratios
%     lambda_ratios(i) = lambda_sorted(i) / lambda_sorted(i+1);
% end

% % --- 7. Print Ratios (Truncated Display) ---
% fprintf('\nSuccessive Eigenvalue Ratios (lambda_i / lambda_i+1) for %dGHz:\n', targetFreq_GHz);
% disp(lambda_ratios)













% % Heatmap Visualization
% % Z_T(logical(eye(size(Z_T)))) = 0;
% Z_T = abs(Z_T);
% fig = figure('visible', 'off');
% imagesc(Z_T);
% colorbar;
% title(['Impedance Matrix Z_T (' num2str(grid_size(1)) 'x' num2str(grid_size(2)) ' elements)']);
% xlabel('Antenna Index (j)');
% ylabel('Antenna Index (i)');
% axis square;
% filename = sprintf('Z_results/%dx%d_spacing%d_%.0fGHz_Z_heatmap.png', ...
%                    grid_size(1), grid_size(2), asf, targetFreq_GHz);
% saveas(fig, filename);
% close(fig);

% fprintf('Successfully saved sorted heatmap to:\n%s\n', filename);

% once you have the true C_T, compute ||  \C_T - \hbC_T ||_F
% \hbC_T is the approximation using the eigencomponents of the impedance matrix
% can try using the first 2, and can try up to and including 1.0183
% i want to be able to see a progression of error
% you may want to plot this vs. # of eigencomponents
% y-axis =  ||  \C_T  - \hbC_T ||_F
% x-axis = # of eigencomponents
% starting with first eigenvalue and go down until the last positive eigenvalue
% can you do that today?
% then plot it until 40
% do the same for C_R

% Carrson C. Fung
% 10:01 AM
% that's not right
% they shouldn't ovelap
% if they are overlapping, then yes, you should make the antenna size smaller so there will not be any overlap

% Carrson C. Fung
% 10:02 AM
% try looking at the matlab tutorial or online to see how to shrink the size of the antennas
% you should be able to change the spacing arbitrary, theoretically to 0 so your array is a continuous aperature
% arbitraily

% Carrson C. Fung
% 10:04 AM
% for XL-MIMO, # of elements in the 1000's,    spacing can be \lambda/2 or less
% for HMIMO, spacing << \lambda/2
% for HMIMO, # of elements = 10K's
% so how many we can simulate depends on memory of our PCs

% Carrson C. Fung
% 10:05 AM
% that's why i said if necessary, I can buy a PC with "lots" of memory to do the sims
% if not, we can try to use the servers at NCHC
% i will just pay for the usage time
% cuz we won't be able to match their compute power

% Carrson C. Fung
% 10:06 AM
% i mean even if I buy a PC with 256GB RAM and the latest server Intel processor, it will never match what NCHC has

% Carrson C. Fung
% 10:08 AM
% i can't read the article