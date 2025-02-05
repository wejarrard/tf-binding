import os
import sys
import shutil

def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <input_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    metadata_dir = 'metadata'

    # Create metadata directory if it doesn't exist and clear its contents
    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)
    os.makedirs(metadata_dir)

    # Process each .txt file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            cell_line_name = os.path.splitext(filename)[0]
            txt_file_path = os.path.join(input_dir, filename)
            tsv_file_path = os.path.join(metadata_dir, f"{cell_line_name}.tsv")
            
            with open(txt_file_path, 'r') as txt_file, open(tsv_file_path, 'w') as tsv_file:
                # Write the header
                tsv_file.write('SAMPLE_ID\tFASTQ_1\tFASTQ_2\n')
                for line in txt_file:
                    srr_id = line.strip()
                    if srr_id:
                        fastq_1 = f"{srr_id}_1"
                        fastq_2 = f"{srr_id}_2"
                        # Write the row
                        tsv_file.write(f"{srr_id}\t{fastq_1}\t{fastq_2}\n")

if __name__ == '__main__':
    main()