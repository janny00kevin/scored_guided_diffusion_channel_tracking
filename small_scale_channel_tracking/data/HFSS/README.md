## Simulation and Data Processing Workflow

1. **HFSS Simulation**: Follow the instruction to simulate the impedance matrix within **HFSS**.
2. **Export & Format Conversion**: Export the impedance matrix as a `.m` file to the `Z_result/` directory. Run `convert_Z_m_to_mat.py` to convert the data into a standard `.mat` file.
3. **Eigenvalue Decomposition**: Execute `run_gevd.py` to perform the Generalized Eigenvalue Decomposition (GEVD). The resulting eigen-components will be saved to `eigen_result/`.
4. **Modal Analysis**: Run `plot_miso_smallest_vs_largest.py` to plot the smallest and largest eigenvalue (EV) modes against the full 64-port configuration and $R_{coupling}$.
5. **Determine $r_T$**: Run `r_T_determining_cum_energy.py` to determine the value of $r_T$ by the cumulated energy based solely on the magnitudes of the eigenvalues.