import os
import logging
from typing import Dict, Optional

def validate_file_exists(path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(path)

def validate_file_not_empty(path: str) -> bool:
    """
    Checks if a file exists and is not empty.
    
    Args:
        path: Path to the file to check
        
    Returns:
        True if the file exists and is not empty, False otherwise
    """
    return os.path.exists(path) and os.path.getsize(path) > 0

def check_cell_lines_in_chip(chip_input_dir: str, cell_lines: Dict[str, str], 
                             aligned_chip_data_dir: str) -> int:
    """
    Get count of valid cell lines from directory structure,
    checking that both the ChIP files exist and have corresponding ATAC files.
    
    Args:
        chip_input_dir: Directory containing ChIP files
        cell_lines: Dictionary mapping cell line keys to cell line names
        aligned_chip_data_dir: Directory containing aligned ATAC data
        
    Returns:
        Count of valid cell lines that have both ChIP and ATAC data
    """
    chip_cell_lines = {
        filename.split("_")[0]
        for filename in os.listdir(chip_input_dir)
        if filename.endswith(".bed")
    }
    
    # Count valid cell lines with ATAC files
    usable_cell_lines = {
        cell_line for cell_line in chip_cell_lines & set(cell_lines.keys())
        if validate_file_not_empty(os.path.join(
            aligned_chip_data_dir, cell_lines[cell_line], "peaks", 
            f"{cell_lines[cell_line]}.filtered.broadPeak"
        ))
    }
    
    return len(usable_cell_lines)

def log_cell_line_status(chip_input_dir: str, cell_lines: Dict[str, str], 
                         aligned_chip_data_dir: str) -> None:
    """
    Log information about available ChIP cell lines and their status.
    
    Args:
        chip_input_dir: Directory containing ChIP files
        cell_lines: Dictionary mapping cell line keys to cell line names
        aligned_chip_data_dir: Directory containing aligned ATAC data
    """
    chip_cell_lines = {
        filename.split("_")[0]
        for filename in os.listdir(chip_input_dir)
        if filename.endswith(".bed")
    }
    
    # Show all available ChIP files
    logging.info("\nAll available ChIP cell lines in directory:")
    for cell_line in sorted(chip_cell_lines):
        if cell_line in cell_lines:
            atac_path = os.path.join(
                aligned_chip_data_dir, cell_lines[cell_line], "peaks", 
                f"{cell_lines[cell_line]}.filtered.broadPeak"
            )
            if validate_file_not_empty(atac_path):
                status = "✓ Will be used"
            else:
                if validate_file_exists(atac_path):
                    status = "✗ Skipped (ATAC file exists but is empty in provided aligned_chip_data_dir)"
                else:
                    status = "✗ Skipped (ATAC file not found in provided aligned_chip_data_dir)"
        else:
            status = "✗ Skipped (not in cell_line_mapping.json)"
        logging.info(f"  - {cell_line}: {status}")
    
    usable_cell_lines = check_cell_lines_in_chip(chip_input_dir, cell_lines, aligned_chip_data_dir)

    logging.info("\n" + "="*50)
    logging.info(f"Total ChIP files: {len(chip_cell_lines)}")
    logging.info(f"Will process: {usable_cell_lines} cell lines")
    logging.info(f"Will skip: {len(chip_cell_lines) - usable_cell_lines} cell lines")
    logging.info("\n" + "="*50) 