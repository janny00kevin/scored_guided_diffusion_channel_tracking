%% UPA_patch.m
clear; clc;

% --- Configuration ---
% freqs = [2.065e9, 38.75e9];
% freqs = [2.065e9];
freqs = [38.75e9];
grid_size = [3, 3];    % 7x7 UPA
asf = 0.5;           % Antenna Spacing Factor (relative to lambda)

% Ensure directories exist
if ~exist('S_results', 'dir'), mkdir('S_results'); end
if ~exist('Z_results', 'dir'), mkdir('Z_results'); end
if ~exist('plot_antenna_layout', 'dir'), mkdir('plot_antenna_layout'); end

for i = 1:length(freqs)
    targetFreq = freqs(i);
    lambda = 3e8 / targetFreq;
    spacing = lambda / asf;
    
    fprintf('\n------------------------------------------------\n');
    fprintf('Designing UPA for %.3f GHz...\n', targetFreq/1e9);
    
    % --- Step 1: Design the Element ---
    p_element = patchMicrostrip;
    % Force FR4 (er=4.4). The higher dielectric constant physically shrinks 
    % the patch, ensuring the metal easily fits inside a lambda/2 grid.
    p_element.Substrate = dielectric('FR4'); 
    p_element = design(p_element, targetFreq);
    
    % --- Step 2: Prevent Intersections (The Fix) ---
    % Restrict the substrate/ground to be marginally smaller than the spacing
    % This stops the 3D solver from detecting overlapping solid volumes.
    p_element.GroundPlaneLength = spacing * 0.99;
    p_element.GroundPlaneWidth  = spacing * 0.99;
    
    % --- Step 3: Construct the Array ---
    upa = rectangularArray('Element', p_element, 'Size', grid_size);
    upa.RowSpacing = spacing;
    upa.ColumnSpacing = spacing;
    
    % --- Step 4: Simulate ---
    fprintf('Calculating S-parameters...\n');
    % mesh(upa, 'MaxEdgeLength', lambda / 10);
    
    tic;
    S_obj = sparameters(upa, targetFreq);
    totalTime = toc;
    fprintf('Simulation took %.2f minutes.\n', totalTime/60);
    
    % --- Step 5: Convert and Save ---
    S_matrix = S_obj.Parameters;
    Z_matrix = s2z(S_matrix, S_obj.Impedance);
    
    % Print Z11 to verify the real part is safely positive
    fprintf('Z11 = %.2f %+.2fi Ohm\n', real(Z_matrix(1,1)), imag(Z_matrix(1,1)));
    
    save(sprintf('S_results/7x7_UPA_%.3fGHz_S.mat', targetFreq/1e9), 'S_matrix', 'targetFreq', 'grid_size');
    save(sprintf('Z_results/7x7_UPA_%.3fGHz_Z.mat', targetFreq/1e9), 'Z_matrix', 'targetFreq', 'grid_size');
end

fprintf('\nCompleted fully automated data generation for all frequencies.\n');