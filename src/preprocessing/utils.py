import os
from typing import List

CELL_LINES = [
    # "NCI-H660",
    "22Rv1",
    # "LNCAP",
    # "PC-3",
    # "C42B",
    # "C4-2",
    # "MCF7",
    # "Ramos",
    "A549",
    # "HT-1376",
    # "K-562",
    # "JURKAT",
    # "Hep_G2",
    # "MCF_10A",
    # "SCaBER",
    # "SEM",
    "786-O",
    # "Ishikawa",
    # "MOLT-4",
    # "BJ_hTERT",
    # "SIHA",
    # "Detroit_562",
    # "OVCAR-8",
    # "PANC-1",
    # "NCI-H69",
    # "HELA",
    # "HuH-7",
    # "THP-1",
    # "K-562",
    # "U-87_MG",
    # "SK-N-SH",
    # "TC-32",
    # "RS411",
    # "TTC1240",
]


def get_cell_line_labels(cell_lines_directory: str) -> List[str]:
    assert os.path.exists(
        cell_lines_directory
    ), f"{cell_lines_directory} does not exist."
    assert os.path.isdir(
        cell_lines_directory
    ), f"{cell_lines_directory} is not a directory."

    found_cell_lines = [
        folder
        for folder in os.listdir(cell_lines_directory)
        if os.path.isdir(os.path.join(cell_lines_directory, folder))
        and os.path.isdir(os.path.join(cell_lines_directory, folder, "peaks"))
        and folder in CELL_LINES
    ]

    # Determine which cell lines from CELL_LINES were not found
    not_found_cell_lines = [
        cell_line for cell_line in CELL_LINES if cell_line not in found_cell_lines
    ]

    # Print the CELL_LINES that were not found
    print(
        f"The following cell lines were not found in the directory: {not_found_cell_lines}"
    )

    return found_cell_lines
