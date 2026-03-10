%% Callable Function Section
function p = createPatchAntenna_2GHz()
    % createPatchAntenna returns a patchMicrostrip object optimized for targetFreq
    % visit the website below to get the antenna config with certain parameters
    % https://3g-aerial.biz/en/online-calculations/antenna-calculations/patch-antenna-online-calculator
    targetFreq = 2.0625e9;
    epsilonR = 4.4;
    height = 0.0014; 
    
    % Object Construction
    p = patchMicrostrip;
    p.Length = 0.03853;        % 38.57 mm (Tuned)
    p.Width = 0.0442;          % 44.2 mm 
    p.Height = height;
    p.Substrate = dielectric('FR4');
    p.Substrate.EpsilonR = epsilonR;
    p.GroundPlaneLength = 0.1455;
    p.GroundPlaneWidth = 0.1455;
    p.FeedOffset = [0.0104 0];
end