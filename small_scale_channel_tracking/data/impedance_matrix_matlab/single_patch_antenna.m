%% Main Execution Block
% This part runs when you execute the script
clear; clc;
% Setup Directories
folders = {'plot_antenna_layout', 'plot_S', 'plot_Z'};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

% 1. Define target frequency
targetFreq = 2.0625e9; % Target frequency in Hz

% 2. Call the function to get the antenna object
p = createPatchAntenna_2GHz();

% 3. Define scan parameters
f_check = linspace(1.95e9, 2.15e9, 21);
mesh(p, 'MaxEdgeLength', 0.008); 
close(gcf); % Close the mesh figure to keep the workspace clean
% fig_layout = figure('visible', 'off');
show(p);
%%
saveas(fig_layout, 'plot_antenna_layout/single_antenna_layout.png');
close(fig_layout);

% 4. Perform Analysis
disp(['Starting final matching optimization for ', num2str(targetFreq/1e9), ' GHz...']);
s = sparameters(p, f_check);

% 5. Plot S11
fig_s11 = figure('visible', 'off');
s11_data = squeeze(20*log10(abs(s.Parameters(1,1,:))));
plot(f_check, s11_data, 'b-', 'LineWidth', 2);
grid on; hold on;
xline(2.0e9, 'g--', '2.0GHz');
xline(2.125e9, 'g--', '2.125GHz');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('S11 Final Verification');
saveas(fig_s11, 'plot_S/single_antenna_S_results.png');
close(fig_s11);

% 6. Plot Impedance
fig_z = figure('visible', 'off');
impedance(p, f_check);
grid on; hold on;
saveas(fig_z, 'plot_Z/single_antenna_impedance_results.png');
close(fig_z);

% 7. Precise Calculation at Resonance
Z_at_resonance = impedance(p, targetFreq);
fprintf('\n------------------------------------------------\n');
fprintf('Impedance at %.2f GHz: %.2f + %.2fj\n', targetFreq/1e9, real(Z_at_resonance), imag(Z_at_resonance));
fprintf('Real Part (Resistance): %.2f Ohm \n', real(Z_at_resonance));
fprintf('Imaginary Part (Reactance): %.2f Ohm \n', imag(Z_at_resonance));
fprintf('------------------------------------------------\n');


