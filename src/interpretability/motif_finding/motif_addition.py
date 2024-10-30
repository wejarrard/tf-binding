#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import pandas as pd
from Bio import motifs
from Bio.Seq import Seq
import pysam
from tqdm import tqdm
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Annotate genomic regions with motif information.")
    parser.add_argument('--tsv_file', required=True, help='Path to the input TSV file.')
    parser.add_argument('--jaspar_file', required=True, help='Path to the JASPAR motif file.')
    parser.add_argument('--reference_genome', required=True, help='Path to the reference genome FASTA file.')
    parser.add_argument('--output_file', required=True, help='Path for the output TSV file.')
    parser.add_argument('--percentile_cutoff', type=float, default=0, help='Percentile cutoff for motif scores.')
    parser.add_argument('--min_score', type=float, default=-10, help='Minimum motif score to retain motifs.')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top motifs to retain.')
    return parser.parse_args()

def add_motifs(tsv_file, jaspar_file, reference_genome, output_file, percentile_cutoff=0, min_score=-10, top_n=10):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Read the motif from the JASPAR file
    logging.info(f"Reading motif from JASPAR file: {jaspar_file}")
    try:
        with open(jaspar_file) as handle:
            motif_collection = motifs.parse(handle, 'jaspar')
            motifs_list = list(motif_collection)
            if len(motifs_list) != 1:
                raise ValueError(f"Expected exactly one motif in {jaspar_file}, found {len(motifs_list)}.")
            motif = motifs_list[0]
    except Exception as e:
        logging.error(f"Failed to read motif from JASPAR file: {e}")
        sys.exit(1)
    
    motif_length = len(motif)
    logging.info(f"Motif length: {motif_length}")
    
    # Create a Position-Specific Scoring Matrix (PSSM)
    pwm = motif.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()
    
    # Open the reference genome FASTA file using pysam
    logging.info(f"Opening reference genome FASTA file: {reference_genome}")
    try:
        fasta = pysam.FastaFile(reference_genome)
    except Exception as e:
        logging.error(f"Failed to open reference genome FASTA file: {e}")
        sys.exit(1)
    
    # Read TSV file into a pandas DataFrame
    logging.info(f"Reading input TSV file: {tsv_file}")
    try:
        df = pd.read_csv(tsv_file, sep='\t')
    except Exception as e:
        logging.error(f"Failed to read TSV file: {e}")
        fasta.close()
        sys.exit(1)
    
    # Validate required columns in the input DataFrame
    required_columns = {'chr', 'start', 'end', 'count', 'label', 'cell_line'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"Input TSV file is missing required columns: {missing}")
        fasta.close()
        sys.exit(1)
    
    # Create empty lists to store results
    all_motifs_rows = []
    best_motifs_rows = []
    top10_motifs_rows = []  # New list for top 10 motifs per region
    scores_collection = []
    
    logging.info("Starting motif processing...")
    # Process each region
    for row in tqdm(df.itertuples(index=False), total=df.shape[0], desc="Processing motifs"):
        chrom = row.chr
        try:
            start = int(row.start) - 1  # 0-based indexing
            end = int(row.end)
        except ValueError:
            logging.warning(f"Invalid start/end positions for row: {row}")
            continue

        # Fetch the sequence from the reference genome
        try:
            seq_str = fasta.fetch(chrom, start, end).upper()
        except Exception as e:
            logging.error(f"Error fetching sequence for {chrom}:{start}-{end} - {e}")
            no_motif_row = {
                'chr': chrom,
                'start': row.start,
                'end': row.end,
                'count': row.count,
                'label': row.label,
                'cell_line': row.cell_line,
                'motif_sequence': 'no_motif',
                'motif_score': np.float16(-np.inf)
            }
            all_motifs_rows.append(no_motif_row)
            best_motifs_rows.append(no_motif_row)
            # For top10, we add the no_motif row once
            top10_motifs_rows.append(no_motif_row)
            continue
        seq = Seq(seq_str)

        # Search for the motif in the sequence with the specified minimum score
        motifs_found = []
        for position, score in pssm.search(seq, threshold=min_score, both=True):
            motif_info = {}
            if position >= 0:
                strand = '+'
                match_seq = str(seq[position:position + motif_length])
                motif_start = start + position
            else:
                strand = '-'
                adjusted_position = -position - 1
                match_seq = str(seq.reverse_complement()[adjusted_position:adjusted_position + motif_length])
                motif_start = end - adjusted_position - motif_length

            # Populate motif information
            motif_info = {
                'chr': chrom,
                'start': row.start,
                'end': row.end,
                'count': row.count,
                'label': row.label,
                'cell_line': row.cell_line,
                'motif_sequence': match_seq,
                'motif_score': np.float16(score)
            }
            motifs_found.append(motif_info)
            all_motifs_rows.append(motif_info)
            scores_collection.append(np.float16(score))
        
        if not motifs_found:
            # No motifs found; append a 'no_motif' row
            no_motif_row = {
                'chr': chrom,
                'start': row.start,
                'end': row.end,
                'count': row.count,
                'label': row.label,
                'cell_line': row.cell_line,
                'motif_sequence': 'no_motif',
                'motif_score': np.float16(-np.inf)
            }
            best_motifs_rows.append(no_motif_row)
            top10_motifs_rows.append(no_motif_row)
        else:
            # Determine the best motif
            best_motif = max(motifs_found, key=lambda x: x['motif_score'])
            best_motifs_rows.append(best_motif)
            
            # Determine top 10 motifs
            sorted_motifs = sorted(motifs_found, key=lambda x: x['motif_score'], reverse=True)
            top10 = sorted_motifs[:top_n]
            top10_motifs_rows.extend(top10)
    
    # Create three new DataFrames: all motifs, best motifs, and top 10 motifs
    all_motifs_df = pd.DataFrame(all_motifs_rows)
    best_motifs_df = pd.DataFrame(best_motifs_rows)
    top10_motifs_df = pd.DataFrame(top10_motifs_rows)
    
    # Calculate percentile threshold
    if scores_collection:
        percentile_threshold = np.float16(np.percentile(scores_collection, percentile_cutoff))
        logging.info(f"Percentile threshold ({percentile_cutoff}%): {percentile_threshold}")
    else:
        percentile_threshold = None
        logging.info("No motif scores available to calculate percentile threshold.")
    
    # Apply the percentile cutoff to all DataFrames if threshold is defined
    if percentile_threshold is not None:
        for df_subset in [all_motifs_df, best_motifs_df, top10_motifs_df]:
            mask = df_subset['motif_score'] < percentile_threshold
            df_subset.loc[mask, 'motif_sequence'] = 'no_motif'
            df_subset.loc[mask, 'motif_score'] = np.float16(-np.inf)
    
    # Define the desired column order
    desired_columns = ['chr', 'start', 'end', 'count', 'label', 'cell_line', 'motif_sequence', 'motif_score']
    
    # Reorder and select the desired columns for all DataFrames
    all_motifs_df = all_motifs_df[desired_columns]
    best_motifs_df = best_motifs_df[desired_columns]
    top10_motifs_df = top10_motifs_df[desired_columns]
    
    # Save the updated DataFrames to output files
    try:
        best_motifs_df.to_csv(output_file, sep='\t', index=False)
        all_output_file = output_file.replace('.csv', '_all_motifs.csv') if output_file.endswith('.csv') else output_file + '_all_motifs.csv'
        top10_output_file = output_file.replace('.csv', '_top10_motifs.csv') if output_file.endswith('.csv') else output_file + '_top10_motifs.csv'
        # all_motifs_df.to_csv(all_output_file, sep='\t', index=False)
        # top10_motifs_df.to_csv(top10_output_file, sep='\t', index=False)
        logging.info(f"Best motifs saved to: {output_file}")
        # logging.info(f"All motifs saved to: {all_output_file}")
        # logging.info(f"Top 10 motifs saved to: {top10_output_file}")
    except Exception as e:
        logging.error(f"Failed to save output files: {e}")
        fasta.close()
        sys.exit(1)
    
    # Close the FASTA file
    fasta.close()
    
    return all_motifs_df, best_motifs_df, top10_motifs_df

if __name__ == '__main__':
    args = parse_arguments()
    add_motifs(
        tsv_file=args.tsv_file,
        jaspar_file=args.jaspar_file,
        reference_genome=args.reference_genome,
        output_file=args.output_file,
        percentile_cutoff=args.percentile_cutoff,
        min_score=args.min_score,
        top_n=args.top_n
    )
