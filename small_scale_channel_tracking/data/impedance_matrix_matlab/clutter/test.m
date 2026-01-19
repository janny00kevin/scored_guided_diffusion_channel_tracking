% % 2. 結合最佳參數
% p = patchMicrostrip;
% p.Length = 0.0344;          % 保持這個長度 (對準 2.07 GHz)
% p.Width = 0.0442;           % 保持寬度
% p.Height = 0.0014;
% p.Substrate = dielectric('FR4');
% p.Substrate.EpsilonR = 4.4;
% p.FeedOffset = [0.009 0];   % 【關鍵修改】從 0.0058 改回 0.009 以加深凹槽

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

opt = optimize()

% 1. Create a standard visible figure (remove 'Visible', 'off')
figure; 
% 2. Display the antenna geometry
show(patchElement); 
% 3. Format the plot
title('Geometry');


%% --- 1. Define Frequency Ranges ---
% A smooth range for the curve (Target +/- 0.05 GHz)
freq_sweep = linspace(targetFreq - 0.05e9, targetFreq + 0.05e9, 11);

% The 3 specific points you asked for
freq_3pts  = [targetFreq - 0.01e9, targetFreq, targetFreq + 0.01e9];

% --- 2. Calculate Data ---
% Calculate for the full sweep
S_sweep = sparameters(patchElement, freq_sweep);
Z_sweep = impedance(patchElement, freq_sweep);

% Calculate for the 3 specific points
S_3pts = sparameters(patchElement, freq_3pts);
Z_3pts = impedance(patchElement, freq_3pts);

% Extract values for plotting
S11_sweep_dB = 20*log10(abs(squeeze(S_sweep.Parameters)));
S11_3pts_dB  = 20*log10(abs(squeeze(S_3pts.Parameters)));

% --- 3. Plot S11 (Return Loss) ---
figure;
plot(freq_sweep/1e9, S11_sweep_dB, 'b-', 'LineWidth', 1.5); hold on;
plot(freq_3pts/1e9,  S11_3pts_dB,  'ro', 'MarkerSize', 8, 'LineWidth', 2);
title('S11 Reflection Coefficient');
xlabel('Frequency (GHz)');
ylabel('Magnitude (dB)');
legend('Frequency Sweep', 'Selected 3 Points');
grid on;

% --- 4. Plot Impedance (Z) ---
figure;
% Plot Resistance (Real part)
plot(freq_sweep/1e9, real(Z_sweep), 'b-', 'LineWidth', 1.5); hold on;
% Plot Reactance (Imaginary part)
plot(freq_sweep/1e9, imag(Z_sweep), 'g--', 'LineWidth', 1.5);

% Add markers for the 3 points
plot(freq_3pts/1e9, real(Z_3pts), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
plot(freq_3pts/1e9, imag(Z_3pts), 'gx', 'MarkerSize', 8, 'LineWidth', 2);

title('Antenna Impedance');
xlabel('Frequency (GHz)');
ylabel('Impedance (Ohms)');
legend('Resistance (Real)', 'Reactance (Imag)', 'R Points', 'X Points');
grid on;


%% 3. 設定掃描參數
f_check = linspace(1.95e9, 2.15e9, 3);
mesh(p, 'MaxEdgeLength', 0.008); % 保持快速計算

% 4. 計算與繪圖
disp('正在進行最終匹配優化...');
s = sparameters(p, f_check);

% 5. 繪圖驗證
fig_geom = figure('Visible', 'off');
s11_data = squeeze(20*log10(abs(s.Parameters(1,1,:))));
plot(f_check, s11_data, 'b-', 'LineWidth', 2);
grid on; hold on;

% 6. 加上輔助標線
yline(-10, 'r--', '合格標準 (-10 dB)');
xline(2.0e9, 'g--', '下限 2.0GHz');
xline(2.125e9, 'g--', '上限 2.125GHz');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('S11 最終驗證：結合最佳長度與偏移');
print(fig_geom, 'spara.png', '-dpng', '-r300');