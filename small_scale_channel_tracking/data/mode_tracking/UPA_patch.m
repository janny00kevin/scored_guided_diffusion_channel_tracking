%% UPA_patch.m
% clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configurations
asf = 2;  % antenna_space = lambda / asf (e.g., lambda/2)
n = 7;
grid_size = [n n]; % [Rows, Columns] 

% 1. Create the base element
if targetFreq > 30e9
    p_element = createPatchAntenna_39GHz();
elseif targetFreq > 20e9
    p_element = createPatchAntenna_28GHz();
else
    p_element = createPatchAntenna_2GHz();
end

% 2. Define Target Frequency and Physics
if ~exist('targetFreq', 'var')
    error('No frequency provided.');
end
c = 299792458;
lambda = c / targetFreq;
fprintf('Antenna Spacing Factor: %d\n', asf);
fprintf('UPA Grid Size: %d x %d (Total: %d elements)\n', ...
    grid_size(1), grid_size(2), prod(grid_size));

% 3. Design the Rectangular (Planar) Array
upa = rectangularArray;
upa.Element = p_element;
upa.Size = grid_size; % Set [Row, Col]
upa.RowSpacing = lambda / asf;    % Spacing along Y-axis usually
upa.ColumnSpacing = lambda / asf; % Spacing along X-axis usually

% A. Resize Ground Plane LENGTH (X-Axis)
num_rows = grid_size(2);
margin_L = (p_element.GroundPlaneLength - p_element.Length) / 2;
boardLen = (lambda/asf * (num_rows-1) + p_element.Length/asf*2 + margin_L*2) / num_rows;
% B. Resize Ground Plane WIDTH (Y-Axis)
num_cols = grid_size(1);
margin_W = (p_element.GroundPlaneWidth - p_element.Width) / 2;
boardWid = (lambda/asf * (num_cols-1) + p_element.Width/asf*2 + margin_W*2) / num_cols;
% C. Apply to the Shared Element
upa.Element.Substrate.Length = boardLen;
upa.Element.GroundPlaneLength = boardLen;
upa.Element.Substrate.Width = boardWid;
upa.Element.GroundPlaneWidth = boardWid;

% Save Layout to verify spacing
if ~exist('plot_antenna_layout', 'dir'), mkdir('plot_antenna_layout'); end
fig = figure('visible', 'off');
show(upa);
title(['UPA Layout (' num2str(grid_size(1)) 'x' num2str(grid_size(2)) ...
       ', Spacing = \lambda/' num2str(asf) ' = ' ...
       num2str(upa.RowSpacing*1000, '%.1f') ' mm)']);
view(45, 45); % Angled view to see the planar structure better
filename = sprintf('plot_antenna_layout/%dx%dUPA_layout_%.2fGHz.png', ...
                   grid_size(1), grid_size(2), targetFreq/1e9);
saveas(fig, filename);
close(fig);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define a reasonable mesh size
meshSize = lambda / 10; 
mesh(upa, 'MaxEdgeLength', meshSize);

% 4. Calculate S-parameters
disp(['Calculating S-parameters for UPA at ', num2str(targetFreq/1e9), ' GHz...']);
tic;
S_obj = sparameters(upa, targetFreq);

% 5. Convert S-parameters to Z-parameters (Impedance Matrix)
Z_matrix = s2z(S_obj.Parameters, S_obj.Impedance);
totalTime = toc;
fprintf('Success! Simulation took %.2f minutes.\n', totalTime/60);

% 6. Print the Z-Matrix
% fprintf('\n------------------------------------------------\n');
% fprintf('Z-Matrix (Impedance Matrix) at %.4f GHz:\n', targetFreq/1e9);
% disp(Z_matrix);
% fprintf('------------------------------------------------\n');

% Extract specific values
% Note: In rectangularArray, indexing usually goes down columns then rows.
% Element 1: (1,1), Element 2: (2,1), Element 3: (1,2), etc.
Z11 = Z_matrix(1,1);
Z12 = Z_matrix(1,2); % Mutual Coupling between El 1 and El 2

fprintf('Self Impedance (Z11):   %.2f %+.2fi Ohm\n', real(Z11), imag(Z11));
fprintf('Mutual Impedance (Z12): %.2f %+.2fi Ohm\n', real(Z12), imag(Z12));

% Save with updated filename format
filename = sprintf('Z_results/%dx%d_UPA_%.2fGHz_Z.mat', ...
    grid_size(1), grid_size(2), targetFreq/1e9);
save(filename, 'Z_matrix', 'targetFreq', 'grid_size');
