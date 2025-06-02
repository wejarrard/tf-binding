# Interpretability Analysis Pipeline

This directory contains scripts for analyzing and interpreting sequence motifs. Follow these steps in order:

1. **Generate Base Data** 
   - Run `interpretability.ipynb` notebook
   - This extracts seqlets (sequence segments) and their attribution scores from the model
   - Runs `python levenstein.py --jaspar motif.jaspar --seqlets positive_seqlets.csv --output lev_pwm.csv`
   - Computes Levenshtein distances between seqlets and known motif PWMs from JASPAR database
   - Outputs similarity scores to `lev_pwm.csv`

2. **Generate Visualization Plots**
   - Run `posthoc.R`
   - Creates various plots analyzing the relationships between:
     - Attribution scores
     - PWM similarities 
     - Seqlet frequencies
   - Outputs plots as PNG files for visualization
