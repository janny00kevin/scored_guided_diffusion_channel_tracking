%% ==================================================
% MIMO LMMSE Channel Estimation (功率歸一化版本)
% 維度規範: [MC, Nt, Nr]
% ==================================================
clear; clc;
SNR_dB = -4;   % 頻道估測環境 SNR (SNR_est)

%% === 1. 載入原始 H 並進行空間總功率歸一化 ===
load('H_UMa_12HZ_rojer.mat', 'H');   % 載入後維度為 [MC, Nt, Nr]
[MC, Nt, Nr] = size(H);

% 【核心修改】：計算真實通道在所有樣本上的「平均總發射功率」
% 也就是每個 [Nt x Nr] 通道矩陣的平均 Frobenius Norm 平方
avg_channel_power = mean(sum(sum(abs(H).^2, 2), 3)); 

% 將真實通道除以平均功率的根號，使其平均總功率嚴格等於 1 (0 dB)
H = H / sqrt(avg_channel_power);

%% === 2. 通道能量統計與雜訊設定 ===
% 此時 H_mean_power 會自動適應歸一化後的尺度 (約等於 1 / (Nt * Nr))
H_mean_power = mean(abs(H(:)).^2);
fprintf('歸一化後通道單一係數平均功率 E{|h|^2} = %.4f\n', H_mean_power);

SNR_linear = 10^(SNR_dB / 10);
% 雜訊功率會隨著歸一化後的通道能量自動連動縮放，確保精準的 -4 dB 估測環境
sigma_w2 = H_mean_power / SNR_linear;
sigma_w  = sqrt(sigma_w2);
fprintf('估計環境 SNR = %d dB, 雜訊功率 sigma_w2 = %.4e\n', ...
        SNR_dB, sigma_w2);

%% === 3. Pilot 設定 ===
P = eye(Nt);       % Unitary pilot
I_Nt = eye(Nt);

%% === 4. 預分配記憶體 ===
H_HAT = complex(zeros(MC, Nt, Nr));

%% === 5. LMMSE Channel Estimation ===
% --- A. 計算傳送端通道協方差矩陣 (Tx-side channel covariance) ---
H_reshaped = reshape(permute(H, [1, 3, 2]), [MC * Nr, Nt]);
CH = (H_reshaped' * H_reshaped) / (MC * Nr);

% --- B. 計算 LMMSE 濾波器矩陣 G ---
G = CH * P' / (P * CH * P' + sigma_w2 * I_Nt);

% --- C. 接收 Pilot 訊號 (加上匹配新尺度的微小雜訊) ---
H_perm = permute(H, [3, 2, 1]);   % 轉換為 [Nr, Nt, MC] 方便矩陣乘法
W = sigma_w * (randn(Nr, Nt, MC) + 1i*randn(Nr, Nt, MC)) / sqrt(2);
Y_perm = pagemtimes(H_perm, P) + W;

% --- D. 執行 LMMSE 估測濾波 ---
H_HAT_perm = pagemtimes(Y_perm, G');   % 結果維度為 [Nr, Nt, MC]

% --- E. 還原成原本的 [MC, Nt, Nr] 維度 ---
H_HAT = permute(H_HAT_perm, [3, 2, 1]);

%% === 6. 計算估計誤差 Delta H ===
% H_DELTA 維度與 H 一致，皆為 [MC, Nt, Nr]
H_DELTA = H - H_HAT; 

%% === 7. 統計檢查與統一打包存檔 ===
H_hat_power = mean(abs(H_HAT(:)).^2);
fprintf('H_HAT 平均單一係數能量 = %.8f (LMMSE 會略小於真實通道)\n', ...
        H_hat_power);

% --- 計算與列印 MSE 與 NMSE ---
mse_error = mean(abs(H_DELTA(:)).^2);
nmse_error = mse_error / H_mean_power;
fprintf('通道估計均方誤差 (MSE) E{|H - H_HAT|^2} = %.8e\n', mse_error);
fprintf('正規化均方誤差 (NMSE) = %.8f\n', nmse_error);

% 打包儲存，確保 Python 讀取的 'H' 是歸一化後的乾淨版本
save('H_HAT_12HZ_rojer.mat', 'H', 'H_HAT', 'H_DELTA'); 
disp('-------------------------------------------------');
fprintf('✅ 已成功儲存功率歸一化後的 H, H_HAT 與 H_DELTA 至 H_HAT_INf.mat\n');