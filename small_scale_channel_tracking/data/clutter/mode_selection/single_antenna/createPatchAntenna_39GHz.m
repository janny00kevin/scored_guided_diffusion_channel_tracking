%% Callable Function Section
function p = createPatchAntenna_2GHz()
    % createPatchAntenna returns a patchMicrostrip object optimized for targetFreq
    % visit the website below to get the antenna config with certain parameters
    % https://3g-aerial.biz/en/online-calculations/antenna-calculations/patch-antenna-online-calculator
    
    targetFreq = 38.75e9;  % 38750 MHz
    epsilonR = 4.4;
    height = 0.0014;       % 1.4 mm
    
    % Object Construction
    p = patchMicrostrip;
    p.Length = 0.00106;    % 1.06 mm
    p.Width = 0.0024;      % 2.4 mm
    p.Height = height;
    p.Substrate = dielectric('FR4');
    p.Substrate.EpsilonR = epsilonR;
    p.GroundPlaneLength = 0.0032; % 3.2 mm
    p.GroundPlaneWidth = 0.0045;  % 4.5 mm
    p.FeedOffset = [0.00016 0];
end
