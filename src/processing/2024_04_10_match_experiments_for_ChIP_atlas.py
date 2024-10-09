# identify candidate TFs
fn_tfs = '/data1/projects/human_cistrome/aligned_chip_data/metadata/shrestha_et_al_TF_enrichment.txt'

# read TFs enriched in at least one subtype
# -----------------------------------------
f = open(fn_tfs)
h = f.readline()
tf2line2count = {}
line2atac = {}
for line in f:
    a = line.rstrip('\r\n').split('\t')
    tf2line2count[a[1]] = {}

f.close()

lines = {}
assays = {}

# all TFs with any form of ChIP data in ChIP-atlas
# ------------------------------------------------
fn_experiments = '/data1/datasets_1/human_cistrome/chip-atlas/2024_04_10_ChIP-atlas_experimentList_parsed.tab'
f = open(fn_experiments)
h = f.readline()
for line in f:
    a = line.rstrip('\r\n').split('\t')
    cell_line = a[2].upper()
    tf = a[1].upper()
    if tf != "NA":
        if tf not in tf2line2count:
            tf2line2count[a[1]] = {}

        lines[cell_line] = 1
        if cell_line in tf2line2count[tf]:
            n = tf2line2count[tf][cell_line]
            tf2line2count[tf][cell_line] = n + 1
        else:
            tf2line2count[tf][cell_line] = 1

f.close()

fo = open('/data1/datasets_1/human_cistrome/chip-atlas/cell_lines.txt', 'w')
for cell_line in lines.keys():
    fo.write(cell_line + '\n')

fo.close()

fn_experiments = '/data1/datasets_1/human_cistrome/chip-atlas/2024_04_10_ChIP-atlas_experimentList.tab'
f = open(fn_experiments)
h = f.readline()
for line in f:
    a = line.rstrip('\r\n').split('\t')
    assay = a[2]
    assays[assay] = 1
    if assay == "ATAC-Seq":
        try:
            cell_line_atac = a[5].upper()
        except:
            continue
        if cell_line_atac in line2atac:
            line2atac[cell_line_atac].append(a[0])
        else:
            line2atac[cell_line_atac] = [a[0]]

f.close()

n = 0
for tf in sorted(tf2line2count.keys()):
    lines_with_tf = tf2line2count[tf].keys()
    paired_lines = []
    for cell_line in lines_with_tf:
        if cell_line in line2atac:
            paired_lines.append(cell_line)

    if len(paired_lines) > 3:
        print(tf + '\t' + str(len(paired_lines)) + '\t' + ','.join(paired_lines))
        n += 1

tfs_C42 = []
tfs_22RV1 = []
tfs_LNCAP = []
for tf in sorted(tf2line2count.keys()):
    if "C4-2" in tf2line2count[tf]:
        tfs_C42.append(tf)
    if "22RV1" in tf2line2count[tf]:
        tfs_22RV1.append(tf)
    if "LNCAP" in tf2line2count[tf]:
        tfs_LNCAP.append(tf)

print(",".join(tfs_C42))
print(",".join(tfs_22RV1))
print(",".join(tfs_LNCAP))

print(set(tfs_C42).intersection(tfs_22RV1))

############################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize variables and sets to store unique TFs and cell lines
tf_filtered = []
cell_lines_set = set()

# Filter and collect TFs with more than 3 paired lines
for tf in sorted(tf2line2count.keys()):
    lines_with_tf = tf2line2count[tf].keys()
    paired_lines = []
    for cell_line in lines_with_tf:
        if cell_line in line2atac:  # Check if the cell line has ATAC-seq data
            paired_lines.append(cell_line)
    
    if len(paired_lines) > 3:  # Only consider TFs with more than 3 paired lines
        tf_filtered.append((tf, paired_lines))  # Store TF and corresponding paired lines
        cell_lines_set.update(paired_lines)  # Collect unique cell lines

# Sort cell lines and transcription factors
cell_lines = sorted(cell_lines_set)
tfs = [tf for tf, _ in tf_filtered]

# Create an empty dataframe to store the binary matrix
data = pd.DataFrame(0, index=tfs, columns=cell_lines)

# Populate the dataframe with 1s where the TF is present in the corresponding cell line
for tf, paired_lines in tf_filtered:
    for cell_line in paired_lines:
        data.loc[tf, cell_line] = 1

# Optional: Check the resulting DataFrame to ensure it has the correct values
print(data)

# Plot the binary heatmap with even larger figure size (doubling the last one)
plt.figure(figsize=(80, 60))  # Doubled the figure size for more space

# Adjust heatmap parameters for better readability
sns.heatmap(data, cmap="Blues", cbar=False, linewidths=0.5, linecolor='gray')

# Customize title and labels with the same font size as before
plt.title("Binary Heatmap of TFs and Cell Lines", fontsize=28)
plt.xlabel("Cell Lines", fontsize=20)
plt.ylabel("Transcription Factors", fontsize=20)

# Adjust rotation of x and y ticks for better readability with the same font size
plt.xticks(rotation=90, fontsize=16)
plt.yticks(rotation=0, fontsize=16)

# Adjust the layout to ensure nothing is cut off
plt.tight_layout()

# Show the plot
plt.savefig("ATAC-CHIP-Heatmap.png")


############################################################################################################
import itertools

# Initialize variables and sets to store unique TFs and cell lines
tf_filtered = []
cell_lines_set = set()

# Filter and collect TFs with more than 3 paired lines
for tf in sorted(tf2line2count.keys()):
    lines_with_tf = tf2line2count[tf].keys()
    paired_lines = []
    for cell_line in lines_with_tf:
        if cell_line in line2atac:  # Check if the cell line has ATAC-seq data
            paired_lines.append(cell_line)
    
    if len(paired_lines) > 3:  # Only consider TFs with more than 3 paired lines
        tf_filtered.append((tf, paired_lines))  # Store TF and corresponding paired lines
        cell_lines_set.update(paired_lines)  # Collect unique cell lines

# Sort cell lines and transcription factors
cell_lines = sorted(cell_lines_set)
tfs = [tf for tf, _ in tf_filtered]

# Create a dictionary to store overlapping information for TF pairs
tf_pair_overlap = {}

# Compute the overlap of cell lines for each pair of transcription factors
for (tf1, lines1), (tf2, lines2) in itertools.combinations(tf_filtered, 2):
    overlap = set(lines1) & set(lines2)  # Find the intersection of cell lines
    if len(overlap) > 1:  # Only consider pairs with more than 1 overlapping cell line
        tf_pair_overlap[(tf1, tf2)] = (len(overlap), sorted(overlap))

# Sort the TF pairs by the number of overlapping cell lines in descending order
sorted_tf_pairs = sorted(tf_pair_overlap.items(), key=lambda x: x[1][0], reverse=True)

# Save the ranking of TF pairs into a text file
with open("tf_pair_overlap_ranking.txt", "w") as f:
    f.write("Ranking of TF pairs based on overlapping cell lines (more than 1):\n")
    for i, ((tf1, tf2), (count, overlapping_lines)) in enumerate(sorted_tf_pairs, 1):
        f.write(f"{i}. {tf1} & {tf2} -> {count} overlapping cell lines: {', '.join(overlapping_lines)}\n")

print("Ranking has been saved to 'tf_pair_overlap_ranking.txt'")