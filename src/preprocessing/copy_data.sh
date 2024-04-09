for cell_line in */; do
    source_dir="${cell_line}pileup_mod_log10/"
    target_dir="/data1/home/wjarrard/projects/tf-binding/src/training/data/cell_lines/${cell_line}"

    if [ -d "${source_dir}" ]; then
        mkdir -p "${target_dir}"
        
        # Loop through all files in the source directory
        for file in "${source_dir}"*; do
            filename=$(basename "${file}")
            # Check if the file already exists in the target directory
            if [ ! -f "${target_dir}${filename}" ]; then
                cp "${file}" "${target_dir}${filename}" &
            fi
        done
    fi
done
wait
