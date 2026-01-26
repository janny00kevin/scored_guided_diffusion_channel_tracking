clear;

% Add path to the TS38901 folder containing chan_gen.m
addpath('TS38901');
rng(0);

% --- 1. Configuration ---
f_c = 28;  % 2.0625, 28, or 38.75 GHz
nTxxy = [7 7];
nRxxy = [2 2];
num_samples = 3000;

% General setting for channel generation
loc_BSs = [0; 0; 20000];
loc_UTs = [0; 0; 0];     % set distance between BSs and UT to be around 680 m in 2D [min 35 m in 2D]
ori_BSs = [1 0 0; 0 1 0; 0 0 -1];
ori_UTs = [1 0 0; 0 1 0; 0 0 1];
f_arr = [2 2]; % UPA n UPA
f_LOSProb = 'LOS';
Tx_d_arr = [0.5 0.5]; % antenna spacing in wavelengths
Rx_d_arr = [0.5 0.5];
Tx_antenna_G = 8; % gain
Rx_antenna_G = 8; % 8 for UMa; 5 for InF
f_disable = [0; 0]; % links all enabled
room_size = [120 50 15];  % only for InF scenario % ISD = 500m according to table 7.2-1 for RMa TR38.901v16 
r_clutter = 0.4; % only for InF scenario
h_clutter = 10; % only for InF scenario
f_scenario = "UMa";
Rx_antenna_G = 8;

parfor i = 1:num_samples
    H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter);
    % while abs(H{1}(1)) > 1e-4 || abs(H{1}(1)) < 1e-6
    %     H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter);
    % end
    H_samples(i,:,:) = single(H{1});
end

filename = sprintf('channel_data_%.0fGHz_%dx%dTx_%dx%dRx_%dsamples.mat', ...
    f_c, nTxxy(1), nTxxy(2), nRxxy(1), nRxxy(2), num_samples);
save(filename,'H_samples','-v7.3');