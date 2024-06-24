import pandas as pd
import sys
import os

def main(output_dir):
    # Reloading the data with no header and manually specifying column names based on the presumed structure
    data = pd.read_csv(os.path.join(output_dir, 'filtered_output.tsv'), sep='\t', header=None, names=['TF', 'Cell_Line', 'Experiment'])

    # Grouping the data by cell line and TF, aggregating the experiments into lists
    grouped_data = data.groupby(['Cell_Line', 'TF']).agg({'Experiment': lambda x: ','.join(x)}).reset_index()

    # Displaying the grouped data
    grouped_data.to_csv(os.path.join(output_dir, 'aggregate.tsv'), header=False, sep='\t', index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    main(output_dir)
