%% Calculate_Coupling_Matrix.m
clear; clc;

% --- 1. Configuration (Must match your saved file) ---
% Note: Ensure these match the simulation that generated the file
targetFreq_GHz = 2.0625;  % Note: '%.0f' in your save command rounded 2.0625 to 2
asf = 2;             % Antenna spacing factor
n = 7;
grid_size = [n n]; % [Rows, Columns] 

% --- 2. Load the Z-Matrix ---
% Construct filename matching your specified pattern
filename = sprintf('Z_results/%dx%d_UPA_%.0fGHz_spacing%d_Z.mat', ...
    grid_size(1), grid_size(2), targetFreq_GHz, asf);

if ~isfile(filename)
    error('File not found: %s\nPlease check the "Z_results" folder or filename format.', filename);
end

fprintf('Loading data from: %s\n', filename);
data = load(filename);

% Extract Z_T (Impedance Matrix)
if isfield(data, 'Z_matrix')
    Z_T = data.Z_matrix;
else
    error('The .mat file does not contain a variable named "Z_matrix".');
end

% --- 3. Implement the Equation ---
Z0 = 50;                  % Characteristic Impedance (Ohm)
[NT, ~] = size(Z_T);      % Number of antennas (e.g., 4 for 2x2)
I_NT = eye(NT);           % Identity Matrix (N_T x N_T)

% Define the denominator matrix A = (Z_T + Z0 * I)
Term_A = Z_T + (Z0 * I_NT);

% Calculate the Inner Product using solvers instead of inversion
% A \ B = A^-1 B; B / A = B A^-1
Inner_Product = (Term_A \ Z_T) / (Term_A');

% Final Calculation
C_T = 0.5 * real(Inner_Product);

% % --- 4. Display and Visualize Results ---
% fprintf('\n------------------------------------------------\n');
% fprintf('Mutual Coupling Matrix (C_T) Dimensions: %dx%d\n', NT, NT);
% fprintf('------------------------------------------------\n');
% disp(C_T);

% % Optional: Save the Coupling Matrix
% save_name = strrep(filename, '_Z.mat', '_Coupling.mat');
% save(save_name, 'C_T', 'Z_T');
% fprintf('\nCoupling matrix saved to: %s\n', save_name);

% Heatmap Visualization
C_T(logical(eye(size(C_T)))) = 0;
C_T = abs(C_T);
figure;
imagesc(C_T);
colorbar;
title(['Mutual Coupling Matrix C_T (' num2str(NT) ' elements)']);
xlabel('Antenna Index (j)');
ylabel('Antenna Index (i)');
axis square;
