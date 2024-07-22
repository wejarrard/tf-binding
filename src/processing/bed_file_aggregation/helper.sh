#!/bin/bash

# Directory containing the TF files
TF_DIR="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data"

# Function to extract the TFs from the directory name
extract_tfs() {
  local dirname=$1
  IFS='_' read -r tf1 tf2 <<< "$dirname"
  echo "$tf1" "$tf2"
}

# Get the list of TF directories
dirs=()
for dir in "$TF_DIR"/*; do
  if [[ -d "$dir" ]]; then
    dirs+=("$(basename "$dir")")
  fi
done

# Generate all pairs and call colocalization.py
for dir in "${dirs[@]}"; do
    read -r tf1 tf2 <<< "$(extract_tfs "$dir")" 
    echo "processing $tf1 and $tf2"
    python colocalization.py "$tf1" "$tf2"
done
