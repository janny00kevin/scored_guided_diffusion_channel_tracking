

cd /media/commlab/TenTB/home/bobo/scored_guided_diffusion_channel_tracking/small_scale_channel_tracking/data/mode_tracking/

# freqs=(38.65 38.70 38.75 38.80 38.85)
freqs=(38.75)

for freq in "${freqs[@]}"
do
    # Convert GHz to Hz string (e.g., 38.65 -> 38.65e9)
    freq_hz="${freq}e9"
    
    echo "Submitting job for ${freq} GHz..."
    
    # Run MATLAB
    # We set the variable 'targetFreq' BEFORE calling the script
    matlab -nodisplay -nosplash -r "targetFreq=${freq_hz}; UPA_patch; exit;" 
    
    # The '&' at the end runs it in the background (Parallel).
    # If you want sequential (one by one), remove the '&'.
    
    sleep 5 # Small delay to prevent license checkout collisions
done

# Wait for all background jobs to finish
# wait
echo "All frequencies done."
