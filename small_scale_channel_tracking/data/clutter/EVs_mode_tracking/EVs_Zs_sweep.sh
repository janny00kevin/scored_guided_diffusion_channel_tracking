#!/bin/bash -l
#SBATCH -A TRI900393
#SBATCH -J impedence_cal
#SBATCH -p ct112
#SBATCH -n 1 # number of process
#SBATCH -N 1 # Maximum number of nodes to be allocated
#SBATCH --mem=70GB
#SBATCH --switches=1 #maximum count of switches
#SBATCH --time=08:00:00
#SBATCH --output=/home/janny00kevin/experiments/EVs_mode_tracking/logs/job_%j.out
#SBATCH --error=/home/janny00kevin/experiments/EVs_mode_tracking/logs/job_%j.err
#SBATCH --mail-type=END,BEGIN   # Send the mail when the job starts and finishes.
#SBATCH --mail-user=janny00kevin@gmail.com

cd /home/janny00kevin/experiments/EVs_mode_tracking
ml apps/matlab/R2024b
# Define Frequencies (GHz)
freqs=(2.0125 2.0375 2.0625 2.0875 2.1125)

# # Create Output Directory
mkdir -p Z_results

for freq in "${freqs[@]}"
do
    # Convert GHz to Hz string (e.g., 38.65 -> 38.65e9)
    freq_hz="${freq}e9"
    
    echo "Submitting job for ${freq} GHz..."
    
    # Run MATLAB
    # We set the variable 'targetFreq_Hz' BEFORE calling the script
    matlab -nodisplay -nosplash -r "targetFreq_Hz=${freq_hz}; UPA_impedance; exit;" 
    
    # The '&' at the end runs it in the background (Parallel).
    # If you want sequential (one by one), remove the '&'.
    
    sleep 5 # Small delay to prevent license checkout collisions
done

# Wait for all background jobs to finish
wait
echo "All frequencies done."
