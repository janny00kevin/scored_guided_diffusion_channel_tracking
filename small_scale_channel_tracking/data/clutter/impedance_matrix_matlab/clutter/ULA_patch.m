%% ULA_patch.m
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configurations
asf = 2;  % antenna_space = lambda / antenna_spacing_ref(asf)
num_elements = 4;

% 1. Create the base element
% Ensure createPatchAntenna_2GHz.m is a separate file in this folder
p_element = createPatchAntenna_2GHz();

% 2. Define Target Frequency and Physics
targetFreq = 2.0625e9;
c = 299792458;
lambda = c / targetFreq;
% fprintf('Target Frequency: %.4f GHz\n', targetFreq/1e9);
% fprintf('Wavelength (lambda): %.4f m\n', lambda);
fprintf('antenna_spacing_ref: %d, num_elements: %d\n', asf, num_elements);

% 3. Design the Linear Array
ula = linearArray;
ula.Element = p_element;
ula.NumElements = num_elements;
ula.ElementSpacing = lambda / asf; % Set spacing

margin = (p_element.GroundPlaneLength - p_element.Length) / 2;
boardLen = (lambda/asf * (ula.NumElements-1) + p_element.Length/asf*2 + margin*2) / ula.NumElements;
ula.Element.Substrate.Length = boardLen;
ula.Element.GroundPlaneLength = boardLen; 

% Save Layout to verify spacing
if ~exist('plot_antenna_layout', 'dir'), mkdir('plot_antenna_layout'); end
fig = figure('visible', 'off');
show(ula);
title(['2x1 ULA Layout (Spacing = \lambda/' num2str(asf) ' = ' num2str(ula.ElementSpacing*1000, '%.1f') ' mm)']);
saveas(fig, 'plot_antenna_layout/ula_layout.png');
close(fig);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define a reasonable mesh size (lambda/10 is standard for accuracy)
% This prevents the 'Out of Memory' error by capping the triangle count
meshSize = lambda / 10; 
mesh(ula, 'MaxEdgeLength', meshSize);

% 4. Calculate S-parameters
disp(['Calculating S-parameters for 2x1 ULA at ', num2str(targetFreq/1e9), ' GHz...']);
tic;
S_obj = sparameters(ula, targetFreq);

% 5. Convert S-parameters to Z-parameters (Impedance Matrix)
Z_matrix = s2z(S_obj.Parameters, S_obj.Impedance);
totalTime = toc;
fprintf('Success! Simulation took %.2f minutes.\n', totalTime/60);

% 6. Print the Z-Matrix
fprintf('\n------------------------------------------------\n');
fprintf('Z-Matrix (Impedance Matrix) at %.4f GHz:\n', targetFreq/1e9);
disp(Z_matrix);
fprintf('------------------------------------------------\n');

% Extract specific mutual impedance values
Z11 = Z_matrix(1,1);
Z12 = Z_matrix(1,2);

fprintf('Self Impedance (Z11):   %.2f %+.2fi Ohm\n', real(Z11), imag(Z11));
fprintf('Mutual Impedance (Z12): %.2f %+.2fi Ohm\n', real(Z12), imag(Z12));

filename = sprintf('%d_ULA_%.0fGHz_spacing%d_Z.mat', ula.NumElements, targetFreq/1e9, asf);
save(filename, 'Z_matrix', 'targetFreq');
