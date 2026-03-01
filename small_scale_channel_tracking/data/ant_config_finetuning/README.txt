1. Run verify_39GHz_single.m to get the initial S11 plot to see the resonant frequency and depth
2. Run tune_length_39GHz.m to fine-tune the patch length

=== Starting Length Optimization Sweep ===
Testing Length = 2.235 mm...
  -> Peak at 38.875 GHz (S11 = -4.73 dB)
Testing Length = 2.240 mm...
  -> Peak at 38.800 GHz (S11 = -4.78 dB)
Testing Length = 2.245 mm...
  -> Peak at 38.725 GHz (S11 = -4.80 dB)

-> 0.002243 
then run verify_39GHz_single.m again to confirm the final S11 plot with the optimized length

3. Run tune_feed_length_39GHz.m to fine-tune the feed length

=== Starting Feed Offset Optimization Sweep ===
Testing Feed Offset = 0.55 mm...
  -> Peak S11 = -6.62 dB at 39.200 GHz
Testing Feed Offset = 0.65 mm...
  -> Peak S11 = -8.42 dB at 39.700 GHz
Testing Feed Offset = 0.75 mm...
  -> Peak S11 = -10.37 dB at 40.000 GHz
Testing Feed Offset = 0.85 mm...
  -> Peak S11 = -12.54 dB at 40.000 GHz
Testing Feed Offset = 0.95 mm...
  -> Peak S11 = -11.64 dB at 40.000 GHz

-> 0.85 mm
need to tune the length again with the new feed offset

=== Final Length Re-Tuning Sweep ===
Testing Length = 2.370 mm...
  -> Peak at 38.725 GHz (S11 = -15.50 dB)
Testing Length = 2.390 mm...
  -> Peak at 38.450 GHz (S11 = -16.00 dB)
Testing Length = 2.410 mm...
  -> Peak at 38.450 GHz (S11 = -21.72 dB)

-> 0.002368


