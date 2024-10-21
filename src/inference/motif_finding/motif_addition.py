#!/usr/bin/env python3

import sys
import csv
from Bio import motifs
from Bio.Seq import Seq
import pysam

def main(tsv_file, jaspar_file, reference_genome, output_file):
    # Read the motif from the JASPAR file
    with open(jaspar_file) as handle:
        motif = motifs.read(handle, 'jaspar')
    
    motif_length = len(motif)
    
    # Create a Position-Specific Scoring Matrix (PSSM)
    pwm = motif.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()

    # Set a threshold for motif detection
    threshold = float('-inf')  # Consider all matches

    # Open the reference genome FASTA file using pysam
    fasta = pysam.FastaFile(reference_genome)

    # Initialize a dictionary to count motif occurrences
    motif_counts = {}

    # First pass: Collect motif counts
    with open(tsv_file, 'r') as tsv_in:
        reader = csv.DictReader(tsv_in, delimiter='\t')
        for row in reader:
            chrom = row['chr']
            start = int(row['start'])
            end = int(row['end'])
            # Adjust for 0-based indexing
            start_pos = start - 1
            end_pos = end
            # Fetch the sequence from the reference genome
            seq_str = fasta.fetch(chrom, start_pos, end_pos).upper()
            seq = Seq(seq_str)
            max_score = None
            best_match_seq = None
            # Search for the motif in the sequence
            for position, score in pssm.search(seq, threshold=threshold, both=True):
                if position >= 0:
                    match_seq = str(seq[int(position):int(position)+motif_length])
                else:
                    # For negative positions, match is on the reverse complement
                    position = -int(position) - 1
                    match_seq = str(seq.reverse_complement()[position:position+motif_length])
                if (max_score is None) or (score > max_score):
                    max_score = score
                    best_match_seq = match_seq
            # Update motif_counts dictionary
            if best_match_seq is not None:
                motif_counts[best_match_seq] = motif_counts.get(best_match_seq, 0) + 1
            else:
                motif_counts['no_motif'] = motif_counts.get('no_motif', 0) + 1

    # Second pass: Write output file with motif counts
    with open(tsv_file, 'r') as tsv_in, open(output_file, 'w', newline='') as tsv_out:
        reader = csv.DictReader(tsv_in, delimiter='\t')
        # Include 'motif_sequence' and 'motif_total_count' in the output fields
        fieldnames = reader.fieldnames + ['motif_sequence', 'motif_total_count']
        writer = csv.DictWriter(tsv_out, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        # Reset the file pointer to the beginning
        tsv_in.seek(0)
        # Skip the header
        next(tsv_in)

        for row in reader:
            chrom = row['chr']
            start = int(row['start'])
            end = int(row['end'])
            # Adjust for 0-based indexing
            start_pos = start - 1
            end_pos = end
            # Fetch the sequence from the reference genome
            seq_str = fasta.fetch(chrom, start_pos, end_pos).upper()
            seq = Seq(seq_str)
            max_score = None
            best_match_seq = None
            # Search for the motif in the sequence
            for position, score in pssm.search(seq, threshold=threshold, both=True):
                if position >= 0:
                    match_seq = str(seq[int(position):int(position)+motif_length])
                else:
                    # For negative positions, match is on the reverse complement
                    position = -int(position) - 1
                    match_seq = str(seq.reverse_complement()[position:position+motif_length])
                if (max_score is None) or (score > max_score):
                    max_score = score
                    best_match_seq = match_seq
            # Update the row with the motif sequence and score
            if best_match_seq is not None:
                row['motif_sequence'] = best_match_seq
                # Optionally include the score
                # row['motif_score'] = max_score
            else:
                row['motif_sequence'] = 'no_motif'
                # row['motif_score'] = 'NA'
            writer.writerow(row)

    # Close the FASTA file
    fasta.close()

    # Print out the counts of each motif sequence
    print("Motif Sequence Counts:")
    for motif_seq, count in motif_counts.items():
        if count > 1:
            print(f"{motif_seq}: {count}")

if __name__ == '__main__':
    # tsv_file = sys.argv[1]
    # jaspar_file = sys.argv[2]
    # reference_genome = sys.argv[3]
    # output_file = sys.argv[4]

    tsv_file = "/Users/wejarrard/projects/tf-binding/data/data_splits/validation_combined_no_motifs.csv"            # The TSV file with regions
    reference_genome = "/Users/wejarrard/projects/tf-binding/data/genome.fa"      # The reference genome file (FASTA format)
    jaspar_file = "/Users/wejarrard/projects/tf-binding/src/inference/motif_finding/motif.jaspar"      # The motif file (JASPAR format)
    output_file = "/Users/wejarrard/projects/tf-binding/data/data_splits/validation_combined.csv"  # Output TSV file with motif column
    main(tsv_file, jaspar_file, reference_genome, output_file)