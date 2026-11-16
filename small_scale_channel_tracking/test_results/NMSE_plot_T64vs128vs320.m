%% NMSE_plot_T64vs128vs320.m
% Recreate the Matplotlib NMSE plot as an editable MATLAB .fig file.
% The .fig is saved beside the corresponding .png in NMSE_plot_png.

clear;
clc;

%% 1. Configuration (keep these values aligned with the Python script)
FREQ_GHZ = 12;

if FREQ_GHZ == 12
    RHO = 0.1034;
    T = [64, 128, 320];
elseif FREQ_GHZ == 38
    RHO = 0.0001;
    T = [64, 128, 256];
else
    error('Unsupported FREQ_GHZ value: %g', FREQ_GHZ);
end

R_T = 60;
mode = 'modal';  % 'spatial' or 'modal'
targetSNR = 10;

scriptDir = fileparts(mfilename('fullpath'));
rawDir = fullfile(scriptDir, 'NMSE_raw_mats');
outputDir = fullfile(scriptDir, 'NMSE_plot_png');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% 2. File names, legend labels, and line styles
rhoText = sprintf('%.3f', RHO);

blue = [0, 0, 1];
green = [0, 0.501960784313725, 0];  % Matplotlib named color "green"

series = repmat(struct( ...
    'label', '', ...
    'filename', '', ...
    'color', [0, 0, 0], ...
    'marker', 'none', ...
    'lineStyle', '-', ...
    'groupIndex', 0), 1, 6);

lineStyles = {'-', '--', ':'};

for k = 1:3
    if strcmp(mode, 'spatial')
        kfFile = sprintf('NMSE_KF_spatial_T%d_%dGHz_rho%s.mat', ...
            T(k), FREQ_GHZ, rhoText);
        ddimFile = sprintf('NMSE_DDIM_spatial_T%d_%dGHz_rho%s_pg.mat', ...
            T(k), FREQ_GHZ, rhoText);
    elseif strcmp(mode, 'modal')
        kfFile = sprintf('NMSE_KF_rT%d_T%d_%dGHz_rho%s.mat', ...
            R_T, T(k), FREQ_GHZ, rhoText);
        ddimFile = sprintf('NMSE_DDIM_rT%d_T%d_%dGHz_rho%s_pg.mat', ...
            R_T, T(k), FREQ_GHZ, rhoText);
    else
        error('mode must be either ''spatial'' or ''modal''.');
    end

    kfIndex = 2*k - 1;
    ddimIndex = 2*k;

    series(kfIndex).label = sprintf('Kalman Filter (T=%d)', T(k));
    series(kfIndex).filename = kfFile;
    series(kfIndex).color = blue;
    series(kfIndex).marker = 's';
    series(kfIndex).lineStyle = lineStyles{k};
    series(kfIndex).groupIndex = k;

    series(ddimIndex).label = ...
        sprintf('DDIM Tracker w/ varied \\eta_k (T=%d)', T(k));
    series(ddimIndex).filename = ddimFile;
    series(ddimIndex).color = green;
    series(ddimIndex).marker = 'd';
    series(ddimIndex).lineStyle = lineStyles{k};
    series(ddimIndex).groupIndex = k;
end

%% 3. Load and plot
fig = figure( ...
    'Color', 'white', ...
    'Units', 'inches', ...
    'Position', [1, 1, 8, 6], ...
    'PaperPositionMode', 'auto');

ax = axes(fig);
hold(ax, 'on');

plotHandles = gobjects(0);
legendLabels = {};
groupY = cell(1, 3);
lastValidSNRs = [];

for k = 1:numel(series)
    filePath = fullfile(rawDir, series(k).filename);

    if ~isfile(filePath)
        warning('File not found: %s', filePath);
        continue;
    end

    data = load(filePath);
    if ~isfield(data, 'snr_range') || ~isfield(data, 'x0_nmse')
        warning('snr_range or x0_nmse is missing from %s', series(k).filename);
        continue;
    end

    snrs = data.snr_range(:);
    values = data.x0_nmse(:);

    if strcmp(mode, 'modal')
        values = values - 10*log10(R_T);
    end

    h = plot(ax, snrs, values, ...
        'Color', series(k).color, ...
        'Marker', series(k).marker, ...
        'LineStyle', series(k).lineStyle, ...
        'LineWidth', 2, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', series(k).color, ...
        'MarkerEdgeColor', series(k).color);

    plotHandles(end+1) = h; %#ok<SAGROW>
    legendLabels{end+1} = series(k).label; %#ok<SAGROW>
    lastValidSNRs = snrs;

    targetIndex = find(abs(snrs - targetSNR) < 1e-10, 1, 'first');
    if ~isempty(targetIndex)
        groupIndex = series(k).groupIndex;
        groupY{groupIndex}(end+1) = values(targetIndex);
    end
end

if isempty(plotHandles)
    close(fig);
    error('No valid data were found in %s.', rawDir);
end

%% 4. Draw the editable ellipses and T labels at SNR = 10 dB
ellipseWidth = 1.5;
labelColor = [1, 0.2, 0.2];  % Approximate Matplotlib red with alpha = 0.8

for k = 1:3
    yValues = groupY{k};
    if numel(yValues) < 2
        continue;
    end

    yCenter = mean(yValues);
    ellipseHeight = abs(max(yValues) - min(yValues)) + 0.8;

    rectangle(ax, ...
        'Position', [targetSNR - ellipseWidth/2, ...
                     yCenter - ellipseHeight/2, ...
                     ellipseWidth, ellipseHeight], ...
        'Curvature', [1, 1], ...
        'FaceColor', 'none', ...
        'EdgeColor', labelColor, ...
        'LineStyle', '-', ...
        'LineWidth', 1.5);

    text(ax, targetSNR + 1.0, yCenter, sprintf('T=%d', T(k)), ...
        'Color', labelColor, ...
        'FontName', 'Helvetica', ...
        'FontSize', 12, ...
        'FontWeight', 'bold', ...
        'VerticalAlignment', 'middle', ...
        'Interpreter', 'none');
end

%% 5. Match the Matplotlib axes, text, grid, and legend
title(ax, 'Channel Tracking Performance', ...
    'FontName', 'Helvetica', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'Interpreter', 'none');

xlabel(ax, 'SNR (dB)', ...
    'FontName', 'Helvetica', ...
    'FontSize', 14, ...
    'Interpreter', 'none');

ylabel(ax, 'Tracking NMSE (dB)', ...
    'FontName', 'Helvetica', ...
    'FontSize', 14, ...
    'Interpreter', 'none');

set(ax, ...
    'FontName', 'Helvetica', ...
    'FontSize', 10, ...
    'Box', 'on', ...
    'GridLineStyle', '--', ...
    'GridAlpha', 0.7, ...
    'Layer', 'bottom');
grid(ax, 'on');

legend(ax, plotHandles, legendLabels, ...
    'FontName', 'Helvetica', ...
    'FontSize', 12, ...
    'Location', 'southwest', ...
    'Interpreter', 'tex');

if ~isempty(lastValidSNRs)
    xlim(ax, [min(lastValidSNRs) - 0.5, max(lastValidSNRs) + 2.5]);
    xticks(ax, lastValidSNRs);
end

hold(ax, 'off');
drawnow;

%% 6. Save the editable MATLAB figure with the same base name as the PNG
if strcmp(mode, 'spatial')
    saveNameBase = sprintf( ...
        'NMSE_Tracker_%dGHz_spatial_rho%s_T%dvs%dvs%d_pg', ...
        FREQ_GHZ, rhoText, T(1), T(2), T(3));
else
    saveNameBase = sprintf( ...
        'NMSE_Tracker_%dGHz_rT%d_rho%s_T%dvs%dvs%d_pg', ...
        FREQ_GHZ, R_T, rhoText, T(1), T(2), T(3));
end

figPath = fullfile(outputDir, [saveNameBase, '.fig']);
savefig(fig, figPath);

fprintf('[Success] Saved editable figure to:\n%s\n', figPath);
