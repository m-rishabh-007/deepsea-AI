#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(dada2)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop('Usage: run_dada2.R <input_dir> <output_prefix> <output_dir> <mode:single|paired> [maxEE] [truncQ] [pool]')
}

input_dir <- args[[1]]
output_prefix <- args[[2]]
output_dir <- args[[3]]
mode <- args[[4]]
maxEE <- ifelse(length(args) >= 5, as.numeric(args[[5]]), 2)
truncQ <- ifelse(length(args) >= 6, as.numeric(args[[6]]), 2)
pool_method <- ifelse(length(args) >= 7, args[[7]], 'pseudo')

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

cat('DADA2 mode: ', mode, '\n')
cat('maxEE: ', maxEE, ', truncQ: ', truncQ, ', pool: ', pool_method, '\n')

if (mode == 'paired') {
  # Identify R1 / R2 cleaned files
  r1 <- sort(list.files(input_dir, pattern = '_R1.*_clean.fastq.gz$', full.names = TRUE))
  r2 <- sort(list.files(input_dir, pattern = '_R2.*_clean.fastq.gz$', full.names = TRUE))
  if (length(r1) == 0 || length(r1) != length(r2)) {
    stop('Paired mode selected but could not find matching R1/R2 cleaned files')
  }
  sample.names <- gsub('_R1.*_clean.fastq.gz$', '', basename(r1))
  names(r1) <- sample.names
  names(r2) <- sample.names

  cat('Found', length(r1), 'paired-end samples\n')
  cat('Processing real FASTQ data with DADA2...\n')
  
  # REAL DADA2 PROCESSING - NO MORE MOCK DATA!
  
  # Filter and truncate sequences first
  cat('Filtering and truncating sequences...\n')
  filtF <- file.path(output_dir, paste0(sample.names, "_F_filt.fastq.gz"))
  filtR <- file.path(output_dir, paste0(sample.names, "_R_filt.fastq.gz"))
  names(filtF) <- sample.names
  names(filtR) <- sample.names
  
  # Filter sequences with permissive quality filtering for real paired-end data (insert size 149)
  out <- filterAndTrim(r1, filtF, r2, filtR, truncLen=c(120,100),
                       maxN=0, maxEE=c(5,8), truncQ=2, rm.phix=TRUE,
                       compress=TRUE, multithread=TRUE, verbose=TRUE)
  
  # Learn error rates from the filtered data
  cat('Learning error rates from forward reads...\n')
  errF <- learnErrors(filtF, multithread = TRUE, randomize = TRUE, verbose = TRUE)
  cat('Learning error rates from reverse reads...\n')
  errR <- learnErrors(filtR, multithread = TRUE, randomize = TRUE, verbose = TRUE)
  
  # Dereplicate sequences
  cat('Dereplicating sequences...\n')
  derepF <- derepFastq(filtF, verbose = TRUE)
  derepR <- derepFastq(filtR, verbose = TRUE)
  names(derepF) <- sample.names
  names(derepR) <- sample.names
  
  # Apply DADA2 algorithm to infer ASVs
  cat('Inferring ASVs from forward reads...\n')
  dadaF <- dada(derepF, err = errF, multithread = TRUE, pool = pool_method)
  cat('Inferring ASVs from reverse reads...\n')
  dadaR <- dada(derepR, err = errR, multithread = TRUE, pool = pool_method)
  
  # Merge paired reads
  cat('Merging paired reads...\n')
  mergers <- mergePairs(dadaF, derepF, dadaR, derepR, verbose = TRUE)
  
  # Construct sequence table
  cat('Constructing sequence table...\n')
  seqtab <- makeSequenceTable(mergers)
} else if (mode == 'single') {
  fastq_files <- list.files(input_dir, pattern = '_clean.fastq.gz$', full.names = TRUE)
  # Exclude any previously filtered files
  fastq_files <- fastq_files[!grepl('filtered_', basename(fastq_files))]
  if (length(fastq_files) == 0) {
    stop('No cleaned fastq files found with pattern _clean.fastq.gz in ', input_dir)
  }
  cat('Found', length(fastq_files), 'cleaned fastq files\n')
  cat('Files:', fastq_files, '\n')
  
    # REAL DADA2 PROCESSING - NO MORE MOCK DATA!
  cat('Processing real FASTQ data with DADA2...\n')
  
  # Get sample names first
  sample.names <- gsub('_clean.fastq.gz$', '', basename(fastq_files))
  
  # Filter and truncate sequences first
  cat('Filtering and truncating sequences...\n')
  filt <- file.path(output_dir, paste0(sample.names, "_filt.fastq.gz"))
  names(filt) <- sample.names
  
  # Filter sequences with aggressive quality filtering
  out <- filterAndTrim(fastq_files, filt, truncLen=150,
                       maxN=0, maxEE=2, truncQ=2, rm.phix=TRUE,
                       compress=TRUE, multithread=TRUE, verbose=TRUE)
  
  # Learn error rates from the filtered data
  cat('Learning error rates...\n')
  err <- learnErrors(filt, multithread = TRUE, randomize = TRUE, verbose = TRUE)
  
  # Dereplicate sequences
  cat('Dereplicating sequences...\n')
  derep <- derepFastq(filt, verbose = TRUE)
  names(derep) <- sample.names
  
  # Apply DADA2 algorithm to infer ASVs
  cat('Inferring ASVs...\n')
  dada_result <- dada(derep, err = err, multithread = TRUE, pool = pool_method)
  
  # Construct sequence table
  cat('Constructing sequence table...\n')
  seqtab <- makeSequenceTable(dada_result)
  rownames(seqtab) <- sample.names
} else {
  stop('Unknown mode: ', mode)
}

cat('Removing chimeras...\n')
# Remove chimeric sequences
seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus", multithread = TRUE, verbose = TRUE)

# Final processing: extract sequences and counts
seqs <- colnames(seqtab.nochim)
counts <- colSums(seqtab.nochim)

# Create final ASV table with real data
asv_df <- data.frame(
  sequence = seqs, 
  count = counts, 
  stringsAsFactors = FALSE
)

# Remove any ASVs with zero counts
asv_df <- asv_df[asv_df$count > 0, ]

cat('Final ASV count:', nrow(asv_df), '\n')
cat('Total reads retained:', sum(asv_df$count), '\n')

asv_csv <- file.path(output_dir, paste0(output_prefix, '_asv_table.csv'))
write.csv(asv_df, asv_csv, row.names = FALSE)

summary_list <- list(
  mode = mode,
  total_asvs = nrow(asv_df),
  total_reads = sum(counts),
  mean_count = mean(counts),
  median_count = median(counts)
)
summary_json <- file.path(output_dir, paste0(output_prefix, '_summary.json'))
write(jsonlite::toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE), summary_json)

cat('Done. Outputs:\n')
cat(asv_csv, '\n')
cat(summary_json, '\n')
