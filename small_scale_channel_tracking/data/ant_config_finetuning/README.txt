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


    Mode  1: Index 38, Eigenvalue 0.8374
    Mode  2: Index 34, Eigenvalue 0.6932
    Mode  3: Index 32, Eigenvalue 0.6737
    Mode  4: Index 18, Eigenvalue 0.3594
    Mode  5: Index 20, Eigenvalue 0.3892
    Mode  6: Index 39, Eigenvalue 0.9334
    Mode  7: Index 42, Eigenvalue 1.1167
    Mode  8: Index 37, Eigenvalue 0.7830
    Mode  9: Index 21, Eigenvalue 0.3929
    Mode 10: Index 17, Eigenvalue 0.3463
    Mode 11: Index 10, Eigenvalue 0.2672
    Mode 12: Index 26, Eigenvalue 0.5349
    Mode 13: Index  8, Eigenvalue -0.2188
    Mode 14: Index  0, Eigenvalue -0.0100
    Mode 15: Index 25, Eigenvalue 0.5294
    Mode 16: Index 35, Eigenvalue 0.7226
    Mode 17: Index 11, Eigenvalue 0.2734
    Mode 18: Index 36, Eigenvalue 0.7759
    Mode 19: Index 27, Eigenvalue 0.5550
    Mode 20: Index 29, Eigenvalue 0.6275
    Mode 21: Index 41, Eigenvalue 1.0790
    Mode 22: Index 12, Eigenvalue 0.2776
    Mode 23: Index 28, Eigenvalue 0.5565
    Mode 24: Index 30, Eigenvalue 0.6375
    Mode 25: Index  7, Eigenvalue 0.2106
    Mode 26: Index 22, Eigenvalue 0.4245
    Mode 27: Index  6, Eigenvalue 0.1869
    Mode 28: Index  9, Eigenvalue 0.2561
    Mode 29: Index 24, Eigenvalue 0.4973
    Mode 30: Index  3, Eigenvalue 0.0789
    Mode 31: Index 23, Eigenvalue 0.4806
    Mode 32: Index  4, Eigenvalue 0.1002
    Mode 33: Index 16, Eigenvalue -0.3373
    Mode 34: Index 13, Eigenvalue -0.2888
    Mode 35: Index 33, Eigenvalue -0.6834
    Mode 36: Index 14, Eigenvalue 0.2934
    Mode 37: Index  2, Eigenvalue 0.0743
    Mode 38: Index 43, Eigenvalue -1.4270
    Mode 39: Index 15, Eigenvalue 0.3095
    Mode 40: Index  5, Eigenvalue -0.1035
    Mode 41: Index 46, Eigenvalue -7.8450
    Mode 42: Index 19, Eigenvalue -0.3857
    Mode 43: Index 31, Eigenvalue -0.6478
    Mode 44: Index 40, Eigenvalue -0.9814
    Mode 45: Index  1, Eigenvalue 0.0112
    Mode 46: Index 45, Eigenvalue -3.6873
    Mode 47: Index 47, Eigenvalue -12.7056
    Mode 48: Index 44, Eigenvalue -1.7883
    Mode 49: Index 48, Eigenvalue -205.2420

