%% Callable Function Section
function p = createPatchAntenna_28GHz()
    % createPatchAntenna returns a patchMicrostrip object optimized for targetFreq
    % visit the website below to get the antenna config with certain parameters
    % https://3g-aerial.biz/en/online-calculations/antenna-calculations/patch-antenna-online-calculator

    targetFreq = 28e9;  % 28 GHz
    epsilonR = 4.4;
    height = 0.0014; 
    
    % Object Construction
    p = patchMicrostrip;
    p.Length = 0.0018;   % 1.8 mm
    p.Width = 0.0033;    % 3.3 mm
    p.Height = height;
    p.Substrate = dielectric('FR4');
    p.Substrate.EpsilonR = epsilonR;
    p.GroundPlaneLength = 0.0047; % 4.7 mm
    p.GroundPlaneWidth = 0.0062;  % 6.2 mm
    % Feed Offset Calculation: (L/2) - x0 = 0.9 - 0.61 = 0.29 mm
    p.FeedOffset = [0.00029 0];
end