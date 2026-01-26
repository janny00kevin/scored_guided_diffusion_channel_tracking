% 1. Setup Parameters
targetFreq = 2.0625e9;
lambda = 3e8 / targetFreq;
spacing = lambda / 2;
% nx = 2;
% ny = 1;

% 2. Define Element & Substrate
d = dielectric('FR4');
d.EpsilonR = 4.4;
d.Thickness = 0.0014;

patchElement = patchMicrostrip;
patchElement.Substrate = d;
patchElement = design(patchElement, targetFreq);
% patchElement.FeedOffset = [0.009 0];
patchElement = patchMicrostrip;
patchElement.Length = 0.0338;          % 保持這個長度 (對準 2.07 GHz)
patchElement.Width = 0.0442;           % 保持寬度
patchElement.Height = 0.0014;
patchElement.Substrate = dielectric('FR4');
patchElement.Substrate.EpsilonR = 4.4;
patchElement.FeedOffset = [0.009 0];

% 3. Define the Array (ULA Version)
nx = 2;
array = linearArray; % Use the dedicated linear array object
array.Element = patchElement;
array.NumElements = nx;
array.ElementSpacing = spacing;

% --- SUBSTRATE FIX for ULA ---
% For a ULA, we only need to worry about the length along the X-axis
boardLen = ((nx-1) * spacing) + 0.0000; 
boardWid = spacing + 0.0000; % Give it some width even though it's a line

array.Element.Substrate.Length = boardLen;
array.Element.Substrate.Width = boardWid;
array.Element.GroundPlaneLength = boardLen;
array.Element.GroundPlaneWidth = boardWid;

% --- Save Antenna Geometry Image ---
disp('Saving antenna geometry plot...');
% 1. Create an invisible figure
fig_geom = figure('Visible', 'off'); 
% 2. Use the 'show' command on the hidden figure
show(array); 
% 3. Format the plot (Optional)
title(sprintf(' Geometry'));
% 4. Save to PNG using the 'print' command
% -dpng: format, -r150: resolution (DPI)
print(fig_geom, 'antenna_geometry.png', '-dpng', '-r150');
% 5. Close the figure to free memory
close(fig_geom);
disp('Geometry saved as antenna_geometry.png');

% --- MESH OPTIMIZATION ---
% Define a reasonable mesh size (lambda/10 is standard for accuracy)
meshSize = lambda / 10; 

% Force the mesh to be simpler
% This prevents the 'Out of Memory' error by capping the triangle count
mesh(array, 'MaxEdgeLength', meshSize);

% 4. Run Simulation
fprintf('Starting %dx%d Simulation...\n', nx, ny);
tic;
S_obj = sparameters(array, targetFreq);
Z_matrix = s2z(S_obj.Parameters, S_obj.Impedance);
totalTime = toc;

% 5. Display and Save Data
fprintf('Success! Simulation took %.2f minutes.\n', totalTime/60);
disp('First 4x4 block of Z-Matrix:');
disp(Z_matrix(1:2, 1:2));

% Save the full matrix and the array object to a file
filename = sprintf('%dx%d_UPA_%.2fGHz_Z.mat', nx, ny, targetFreq/1e9);
save(filename, 'Z_matrix', 'array', 'targetFreq');
disp('Full results saved to antenna_results.mat');