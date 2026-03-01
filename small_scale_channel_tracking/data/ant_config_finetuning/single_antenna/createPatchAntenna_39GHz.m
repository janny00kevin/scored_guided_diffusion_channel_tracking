%% Callable Function Section
function p = createPatchAntenna_39GHz()
    % createPatchAntenna returns a patchMicrostrip object optimized for targetFreq
    % visit the website below to get the antenna config with certain parameters
    % https://3g-aerial.biz/en/online-calculations/antenna-calculations/patch-antenna-online-calculator
    
    targetFreq = 38.75e9;  % 38750 MHz
    epsilonR = 2.2;        % Updated from 4.4 to 2.2
    height = 0.0005;       % 0.5 mm
    
    % Object Construction
    p = patchMicrostrip;
    % Dimensions from calculation
    p.Length = 0.002381;     
    p.Width = 0.0031;      % W = 3.1 mm
    p.Height = height;     % h = 0.5 mm
    % Substrate Configuration
    p.Substrate = dielectric('Name', 'Custom_Substrate');
    p.Substrate.EpsilonR = epsilonR;
    p.Substrate.Thickness = height;
    % Ground Plane Dimensions
    p.GroundPlaneLength = 0.005;  % eL = 5.0 mm
    p.GroundPlaneWidth = 0.0058;  % eW = 5.8 mm
    % Feed Offset Calculation
    % MATLAB FeedOffset is [x y] from the center of the patch.
    % To match 50 Ohm at x0 = 0.7mm from the edge:
    % Offset = (Length/2) - x0 => (2.3/2) - 0.7 = 0.45 mm
    p.FeedOffset = [0.00090, 0];
end
