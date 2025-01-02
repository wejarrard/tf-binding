suppressPackageStartupMessages( require(optparse, quietly=TRUE, warn.conflicts=FALSE) )
suppressPackageStartupMessages( require(ATACseqQC, quietly=TRUE, warn.conflicts=FALSE) )
suppressPackageStartupMessages( require(TxDb.Hsapiens.UCSC.hg38.knownGene, quietly=TRUE, warn.conflicts=FALSE) )
suppressPackageStartupMessages( require(Rsamtools, quietly=TRUE, warn.conflicts=FALSE) )

option_list <- list(
  make_option(c("-i", "--input"), type = "character", help = "path to BAM file of aligned ATAC-seq data [Required]"),
  make_option(c("-o", "--output"), type = "character", help = "path to QC text file to write [Required]"),
  make_option(c("-v", "--verbose"), type = "character", action="store_true", help = "verbose output")  
)

parseobj = OptionParser(option_list=option_list, usage = "usage: Rscript %prog [options]")
opt = parse_args(parseobj)
fn_bam_in = opt$input
fn_output = opt$output
verbose = opt$verbose

if(verbose){
	print(paste("Reading BAM", fn_bam_in))
}
if( !file.exists( fn_bam_in ) ){
	stop( paste("BAM file",fn_bam_in,"does not exist") )
}
txs <- transcripts(TxDb.Hsapiens.UCSC.hg38.knownGene)
## fn_bam_in tags to be read in
possibleTag <- list("integer"=c("AM", "AS", "CM", "CP", "FI", "H0", "H1", "H2", 
                                "HI", "IH", "MQ", "NH", "NM", "OP", "PQ", "SM",
                                "TC", "UQ"), 
                 "character"=c("BC", "BQ", "BZ", "CB", "CC", "CO", "CQ", "CR",
                               "CS", "CT", "CY", "E2", "FS", "LB", "MC", "MD",
                               "MI", "OA", "OC", "OQ", "OX", "PG", "PT", "PU",
                               "Q2", "QT", "QX", "R2", "RG", "RX", "SA", "TS",
                               "U2"))

#sample_id="SRX8746239"
#dir_root="/data1/projects/human_cistrome/aligned_chip_data/SRA"
#fn_bam_in = paste0(dir_root,"/",sample_id,"/alignment/",sample_id,".bowtie.sorted.nodup.bam")
bamTop100 = scanBam(BamFile(fn_bam_in, yieldSize = 100),
                     param = ScanBamParam(tag=unlist(possibleTag)))[[1]]$tag
tags = names(bamTop100)[lengths(bamTop100)>0]
seqinformation = seqinfo(TxDb.Hsapiens.UCSC.hg38.knownGene)
seqlevel = c( paste0( "chr", 1:22), "chrX", "chrY")
which = as(seqinformation[ seqlevel ], "GRanges")
bam = readBamFile(fn_bam_in, tag=tags, which=which, asMates=FALSE, bigFile=TRUE)
if(verbose){
	print("Calculating TSSE...")
}
tsse = TSSEscore( bam, txs )
df = data.frame( tsse$values, row.names=paste0("nt_",100*(-9:10-.5)))
df = rbind( data.frame( tsse.values = tsse$TSSEscore),  df )
dimnames(df)[[1]][ 1 ] = "TSSE"
df = round(df, 4)
write.table( df, file=fn_output, col.names=FALSE, quote=FALSE )
if(verbose){
	print(paste("Wrote output to", fn_output) )
}