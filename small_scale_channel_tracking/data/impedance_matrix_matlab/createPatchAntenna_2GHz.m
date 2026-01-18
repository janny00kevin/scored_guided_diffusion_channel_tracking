%% Callable Function Section
function p = createPatchAntenna_2GHz()
    % createPatchAntenna returns a patchMicrostrip object optimized for targetFreq
    % visit the website below to get the antenna config with certain parameters
    % https://3g-aerial.biz/en/online-calculations/antenna-calculations/patch-antenna-online-calculator
    targetFreq = 2.0625e9;
    epsilonR = 4.4;
    height = 0.0014; 
    
    % Object Construction adjusted for 1.965 GHz
    p = patchMicrostrip;
    p.Length = 0.0361;    
    p.Width = 0.0464;  
    p.Height = height;
    p.Substrate = dielectric('FR4');
    p.Substrate.EpsilonR = epsilonR;
    p.GroundPlaneLength = 0.0735;
    p.GroundPlaneWidth = 0.0838;
    p.FeedOffset = [0.00605 0];   % 0.0058
end