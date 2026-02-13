Use logically identical 2 files: UPA_patch / UPA_impedance computed on 72 / NCHC, but the 2 Zs are quite different,
Therefore, try to run UPA_Patch on NCHC, but it seems the the result is identiical to the UPA_impedance ran on NCHC.
Then move the UPA_patch files from NCHC back to 72 run again to see the result.
- result: The result run on this machine(72) Z_matrix(1,1) = 1.1267e+03 + 4.7516e+02i, which is so different on NCHC: Z_matrix(1,1) = 1.0187e+02 - 2.9141e+02i

1. Run 2GHz_Z_val.sh/39GHz_Z_val.sh to get the impedance matrices at different frenquencies 
    (freqs=(2.0125 2.0375 2.0625 2.0875 2.1125) / freqs=(38.65 38.70 38.75 38.80 38.85)) 
    in the sh file:
    matlab UPA_impedance calculate the impedance matrices for 5 freqs
2. run Generalized_Eig_Decomp.m to GEVD the Zs for 5 freqs (manually modify freqs_GHz needed)
3. run mode_tracking.py to compute eigenvectors correlation to track the eigenvalues for 5 adjencent freqs
    plot the eigenvalues varying along different freqs (manually modify freqs_GHz needed)
- The results (Mode_Tracking_All_Modes.png) shows that they are all flat, only 1 mode across 0 (39GHz).