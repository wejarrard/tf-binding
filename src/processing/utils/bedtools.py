import os
import subprocess
import pandas as pd
import tempfile



def subtract_bed_files(df, regions_to_subtract_df):
    """
    Subtracts regions in the regions_to_subtract_df from the df using bedtools.

    Parameters:
    df (pd.DataFrame): DataFrame containing the main regions with columns ['chr', 'start', 'end'].
    regions_to_subtract_df (pd.DataFrame): DataFrame containing regions to subtract with columns ['chr', 'start', 'end'].

    Returns:
    pd.DataFrame: DataFrame containing regions in df after subtraction of regions_to_subtract_df regions.
    """
    
    column_names = df.columns.tolist()

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_df_bed, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_regions_bed, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_subtracted_bed:
        
        temp_df_bed_file = temp_df_bed.name
        temp_regions_bed_file = temp_regions_bed.name
        temp_subtracted_bed_file = temp_subtracted_bed.name
        
        # Save dataframes to temporary files
        df.to_csv(temp_df_bed_file, sep="\t", index=False, header=False)
        regions_to_subtract_df.to_csv(temp_regions_bed_file, sep="\t", index=False, header=False)

        # Construct the command to subtract the bed files
        command = f"bedtools subtract -a {temp_df_bed_file} -b {temp_regions_bed_file} -A > {temp_subtracted_bed_file}"

        # Execute the command using subprocess
        subprocess.run(command, shell=True, check=True)
        
        # Read the subtracted bed file into a DataFrame
        subtracted_df = pd.read_csv(temp_subtracted_bed_file, sep="\t", header=None, usecols=[i for i in range(len(column_names))], names=column_names)

    # Remove the temporary files
    os.remove(temp_df_bed_file)
    os.remove(temp_regions_bed_file)
    os.remove(temp_subtracted_bed_file)

    return subtracted_df

def intersect_bed_files(df, regions_to_intersect_df, region_type=None):
    """
    Intersects regions in the regions_to_intersect_df with the df using bedtools.

    Parameters:
    df (pd.DataFrame): DataFrame containing the main regions with columns ['chr', 'start', 'end', *]. will keep all columns in this DataFrame.
    regions_to_intersect_df (pd.DataFrame): DataFrame containing regions to intersect with columns ['chr', 'start', 'end'], will drop all columns in this DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing intersected regions between df and regions_to_intersect_df, including all columns from both DataFrames.
    """
    
    column_names = df.columns.tolist()

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_df_bed, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_regions_bed, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_intersected_bed:
        
        temp_df_bed_file = temp_df_bed.name
        temp_regions_bed_file = temp_regions_bed.name
        temp_intersected_bed_file = temp_intersected_bed.name
        
        # Save dataframes to temporary files
        df.to_csv(temp_df_bed_file, sep="\t", index=False, header=False)
        regions_to_intersect_df.to_csv(temp_regions_bed_file, sep="\t", index=False, header=False)

        # Construct the command to intersect the bed files with all columns retained
        command = f"bedtools intersect -a {temp_df_bed_file} -b {temp_regions_bed_file} -wa -wb > {temp_intersected_bed_file}"

        # Execute the command using subprocess
        subprocess.run(command, shell=True, check=True)
        
        # Read the intersected bed file into a DataFrame
        intersected_df = pd.read_csv(temp_intersected_bed_file, sep="\t", header=None, usecols=[i for i in range(len(column_names))], names=column_names)

    intersected_df = intersected_df.drop_duplicates().sort_values(by=['chr', 'start', 'end']).reset_index(drop=True)

    # Add region type
    if region_type is not None:
        intersected_df['region_type'] = region_type

    # Remove the temporary files
    os.remove(temp_df_bed_file)
    os.remove(temp_regions_bed_file)
    os.remove(temp_intersected_bed_file)


    return intersected_df