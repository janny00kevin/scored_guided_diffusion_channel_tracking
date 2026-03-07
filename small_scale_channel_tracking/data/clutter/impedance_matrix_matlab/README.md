# Impedance Matrix Analysis & Generalized Eigenvalue Decomposition

This project analyzes the "effective rank" of antenna arrays at different frequencies (e.g., 2 GHz, 28 GHz, 39 GHz) by examining the Generalized Eigenvalue Decomposition (GEVD) of the impedance matrix $\mathbf{Z}$.

## Workflow

### 1. Generate Impedance Matrix ($\mathbf{Z}$)
**Script:** `UPA_patch.m`

This script simulates a Uniform Planar Array (UPA) using the MATLAB Antenna Toolbox and calculates its Z-parameters.

**Steps:**
1.  Open `UPA_patch.m`.
2.  Set the `targetFreq` variable (e.g., `2.0625e9`, `28e9`, or `38.75e9`).
3.  **Crucial:** Uncomment/Call the corresponding antenna design function for your frequency:
    * For **2 GHz**: Use `p_element = createPatchAntenna_2GHz();`
    * For **28 GHz**: Use `p_element = createPatchAntenna_28GHz();`
    * For **39 GHz**: Use `p_element = createPatchAntenna_39GHz();`
4.  Run the script.
5.  **Output:** A `.mat` file (e.g., `Z_results/7x7_UPA_28GHz_spacing2_Z.mat`) containing the complex impedance matrix $\mathbf{Z}$.

---

### 2. Compute Generalized Eigendecomposition (GEVD)
**Script:** `Generalized_Eig_Decomp.m`

This script decomposes the real and imaginary parts of the impedance matrix to analyze the array's coupling modes.

**Steps:**
1.  Open `Generalized_Eig_Decomp.m`.
2.  Update the **Configuration** section (`targetFreq_GHz`) to match the `.mat` file generated in Step 1.
3.  Run the script.
4.  **Output:** Saves sorted eigenvalues and eigenvectors to `eigen_result/`.

#### Mathematical Formulation
The impedance matrix is defined as:

$$
\mathbf{Z} = \mathbf{R} + j\mathbf{X}
$$

where $\mathbf{R}$ is the resistance matrix (real part) and $\mathbf{X}$ is the reactance matrix (imaginary part).

To analyze the independent modes, we solve the **Generalized Eigenvalue Problem** where we diagonalize $\mathbf{R}_T$ and $\mathbf{X}_T$ simultaneously. As shown in the reference figure, the decomposition is:

$$
\mathbf{R}_T \mathbf{U}_T = \mathbf{X}_T \mathbf{U}_T \mathbf{\Lambda}_T
$$

where:
* $\mathbf{U}_T$ contains the generalized eigenvectors.
* $\mathbf{\Lambda}_T$ is the diagonal matrix of generalized eigenvalues.

*Note: In the context of the Sinha's 2025 paper, the "reciprocal" formulation is used.

---

### 3. Analyze Effective Rank (Cumulative Energy)
**Script:** `Plot_Cumulative_ratio.m`

Once you have generated the eigen-data for all three frequencies (2, 28, and 39 GHz), this script plots the cumulative energy ratio to determine the "effective rank" (degrees of freedom) of the channel.

**Steps:**
1.  Ensure all three `_eigen.mat` files exist in the `eigen_result/` folder.
2.  Run `Plot_Cumulative_ratio.m`.
3.  **Output:** Generates comparison plots showing how energy concentrates in the dominant eigenmodes at different frequencies.
    * **Interpretation:** A curve that reaches 90% energy with fewer eigenvalues indicates a lower effective rank (stronger spatial correlation/coupling).

## Folder Structure
* `data/impedance_matrix_matlab/`
    * `Z_results/` - Stores raw $\mathbf{Z}$ matrices.
    * `eigen_result/` - Stores computed eigenvalues and final plots.
    * `createPatchAntenna_*.m` - Antenna geometry definitions.