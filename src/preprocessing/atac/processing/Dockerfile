# Start with a base image that includes R (Debian-based)
FROM rocker/r-ver:4.4

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for R and other tools
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libxt-dev \
    libgit2-dev \
    zlib1g-dev \
    libpng-dev \
    libjpeg-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    build-essential \
    libbz2-dev \
    liblzma-dev \
    libpcre2-dev \
    libncurses-dev \
    libreadline-dev \
    tabix \
    bcftools \
    bedtools \
    curl \
    wget \
    samtools \
    python3 \
    python3-pip \
    libgsl-dev \
    bc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clean up redundant lists to keep the image small
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Bioconductor and R packages
RUN R -e "install.packages('BiocManager', repos='https://cloud.r-project.org')"
# Set Bioconductor version to 3.16
RUN R -e "BiocManager::install(version = '3.19')"
# Install specific packages for Bioconductor 3.16
RUN R -e "BiocManager::install(c('ATACseqQC', 'ChIPQC', 'GenomicAlignments', 'Rsamtools'), version = '3.19')"
RUN R -e "BiocManager::install('BSgenome.Hsapiens.UCSC.hg38', version = '3.19')"
RUN R -e "BiocManager::install('TxDb.Hsapiens.UCSC.hg38.knownGene', version = '3.19')"
RUN R -e "BiocManager::install('BiocParallel', version = '3.19')"

# Install optparse from CRAN
RUN R -e "install.packages('optparse', repos='https://cloud.r-project.org')"

# Install htslib
RUN curl -L https://github.com/samtools/htslib/releases/download/1.17/htslib-1.17.tar.bz2 -o htslib.tar.bz2 && \
    tar -xjf htslib.tar.bz2 && \
    cd htslib-1.17 && \
    ./configure && \
    make && \
    make install

# Clean up
RUN rm -rf /var/lib/apt/lists/* htslib-1.17 htslib.tar.bz2

RUN pip install MACS2 deeptools

# Set up environment paths for easy access to installed tools
ENV SAMTOOLS="/usr/bin/samtools"
ENV MACS2="/usr/local/bin/macs2"
ENV BEDTOOLS="/usr/bin/bedtools"
ENV BGZIP="/usr/bin/bgzip"
ENV TABIX="/usr/bin/tabix"
ENV BAMCOVERAGE="/usr/local/bin/bamCoverage"

# Set the working directory
WORKDIR /home/project

COPY preprocessing.sh /home/project/
RUN chmod +x /home/project/preprocessing.sh


# Set the default command to bash instead of R
ENTRYPOINT ["/bin/bash", "/home/project/preprocessing.sh"]

# docker run -it \
#   --cpus=8 \
#   --memory=32g \
#   -v /data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/purgatory/K562/bam/K562_merge.bam:/data/K562_merge.bam \
#   -v /data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines:/output \
#   atac_bam_merge_downstream TEST "/data/*" "/output"

# docker run -it \
#   --cpus=8 \
#   --memory=32g \
#   -v /data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/purgatory/K562/bam/K562_merge.bam:/data/K562_merge.bam \
#   -v /data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines:/output \
#   --entrypoint /bin/bash atac_bam_merge_downstream