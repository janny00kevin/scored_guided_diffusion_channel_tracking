1. Run EVs_Zs_sweep.sh to get the impedance matrices at different frenquencies (freqs=(2.0125 2.0375 2.0625 2.0875 2.1125)) 
    in the sh file:
    matlab UPA_impedance calculate the impedance matrices for 5 freqs
2. run Generalized_Eig_Decomp.m to GEVD the Zs for 5 freqs
3. run mode_tracking.py to compute eigenvectors correlation to track the eigenvalues for 5 adjencent freqs
    plot the eigenvalues varying along different freqs
- The results (Mode_Tracking_All_Modes.png) shows that they are all flat, only 1 mode across 0.
