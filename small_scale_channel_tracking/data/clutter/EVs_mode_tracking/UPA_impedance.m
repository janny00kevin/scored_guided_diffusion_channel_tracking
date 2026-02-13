% UPA_impedance.m

% 1. Check for input from Bash (passed as workspace variable)
if ~exist('targetFreq_Hz', 'var')
    warning('No frequency provided. Defaulting to 38.75 GHz.');
    targetFreq_Hz = 38.75e9;
end

% 2. Setup Parameters
c = 299792458;
lambda = c / targetFreq_Hz;
asf = 2; % antenna spacing factor
spacing = lambda / asf;
grid_size = [7, 7]; % 7x7 UPA

% 3. Call Antenna Design
% Ensure you add the path to your single_antenna folder
addpath('single_antenna');

if targetFreq_Hz > 30e9
    p_element = createPatchAntenna_39GHz(); 
elseif targetFreq_Hz > 20e9
    p_element = createPatchAntenna_28GHz();
else
    p_element = createPatchAntenna_2GHz();
end

% 4. Define Array
upa = rectangularArray;
upa.Element = p_element;
upa.Size = grid_size;
upa.RowSpacing = spacing;
upa.ColumnSpacing = spacing;

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
                   grid_size(1), grid_size(2), targetFreq_Hz/1e9);
saveas(fig, filename);
close(fig);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define a reasonable mesh size
meshSize = lambda / 10; 
mesh(upa, 'MaxEdgeLength', meshSize);

% 5. Run Simulation
fprintf('Starting Simulation for %.4f GHz...\n', targetFreq_Hz/1e9);
tic;
S_obj = sparameters(upa, targetFreq_Hz);
Z_matrix = s2z(S_obj.Parameters, S_obj.Impedance);
totalTime = toc;
fprintf('Success! Simulation took %.2f minutes.\n', totalTime/60);

% 6. Save with HIGH PRECISION Filename
% Use %.2f to distinguish 38.70 from 38.75
filename = sprintf('Z_results/%dx%d_UPA_%.2fGHz_Z.mat', ...
    grid_size(1), grid_size(2), targetFreq_Hz/1e9);

save(filename, 'Z_matrix', 'targetFreq_Hz', 'grid_size');
fprintf('Saved to: %s\n', filename);