#===============================================================================
#' Description
#' Create shifted bam file from the merge bams files
#' 
#===============================================================================
#' Author: 
#' - Thaidy Moreno-Rodriguez
#===============================================================================
#' Date: 06 Oct 2023
#===============================================================================
#****************************************************************************
#' Used to create a shifted bam file from merged bam (one per cell line)
#****************************************************************************

## load the libraries
# library(BiocManager)
# library(ChIPQC)
# library(BSgenome.Hsapiens.UCSC.hg38)
# library(TxDb.Hsapiens.UCSC.hg38.knownGene)
# library(GenomicAlignments)
# library(Rsamtools)
# library(BiocParallel)

#===============================================================================
suppressPackageStartupMessages(
  require(BiocManager, quietly=TRUE, warn.conflicts=FALSE)
)

suppressPackageStartupMessages(
  require(optparse, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(GenomicRanges, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(Rsubread, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(ATACseqQC, lib.loc = "/opt/R/4.1.2/lib/R/library", quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(BSgenome.Hsapiens.UCSC.hg38, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(TxDb.Hsapiens.UCSC.hg38.knownGene, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(GenomicAlignments, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(Rsamtools, quietly=TRUE, warn.conflicts=FALSE)
)
suppressPackageStartupMessages(
  require(BiocParallel, quietly=TRUE, warn.conflicts=FALSE)
)

library(ATACseqQC)

#===============================================================================
option_list <- list(
  optparse::make_option(c("-s", "--sample_id"), type = "character", help = "sample identifier [Required]"),
  optparse::make_option(c("-d", "--dir_bam"), type = "character", help = "folder where BAM file is located [Required]"),
  optparse::make_option(c("-o", "--dir_out"), type = "character", help = "output folder where BAM file will be written. [Required]"),
  optparse::make_option(c("-c", "--num_cores"), type = "numeric", help = "number of cores to use [Required]")
)
parseobj = OptionParser(option_list=option_list, usage = "usage: Rscript %prog [options]")
opt = parse_args(parseobj)
sample_id = opt$sample_id 
dir_bam = opt$dir_bam
dir_out = opt$dir_out
num_cores = opt$num_cores

#===============================================================================
## input is bamFile
fn_bam_atac = paste(dir_bam, '/', sample_id, '_merge.bam', sep='')

## bamfile tags from the original bam file

tags <- c("MD","PG","XG", "NM","XM", "XN","XO", "AS", "XS", "YS", "YT", "CP")
#===============================================================================
## files will be output into dir_out
if (!dir.exists(dir_out)){
  dir.create(dir_out)
}else{
    print( paste(dir_out, "directory exits") )
}

Sys.setenv(TMPDIR = dir_out)

#===============================================================================
## GAlignmentsLists shift the bam file by the 5'ends.
## All reads aligning to the positive strand will be offset by +4bp, 
## and all reads aligning to the negative strand will be offset -5bp by default.

sliding_window_size <- 5e7

# Read and process the BAM file in chunks
atac_bam <- readBamFile(fn_bam_atac, tag = tags, bigFile = TRUE, asMates = TRUE)

BPPARAM <- MulticoreParam(workers = num_cores, progress=FALSE)

shiftedBamfile <- file.path(dir_out, paste0(sample_id,"_merge.shifted.bam"))

# Shift alignments for each chunk and write to output
shifted_atac_bam <- shiftGAlignmentsList(atac_bam, outbam = shiftedBamfile, slidingWindowSize = sliding_window_size, BPPARAM=BPPARAM)

message("BAM file processing complete. Shifted BAM saved at: ", shiftedBamfile)

#===============================================================================
