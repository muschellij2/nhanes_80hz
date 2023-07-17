#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(agcounts)
suppressPackageStartupMessages(library("optparse"))
source("helper_functions.R")
Sys.unsetenv("RETICULATE_PYTHON")
source("helper_functions.R")
options(digits.secs = 3)
bucket = "nhanes_80hz"
bucket_setup(bucket)
config = trailrun::cr_gce_setup(bucket = bucket)

reticulate::py_config()
reticulate::source_python("signal_nil_check.py")

option_list <- list(
  make_option(c("-v", "--verbose"), action="store_true", default=TRUE,
              help="Print extra output [default]"),
  make_option(c("-i", "--id"), type="integer",
              help="ID to run",
              metavar="number")
)
opt <- parse_args(OptionParser(option_list=option_list))
wide = readr::read_rds("wide.rds")
nfolds = 100
wide = make_folds(wide, nfolds)
stopifnot(!is.null(opt$id))
idf = wide[ wide$id %in% opt$id,]
print(idf)
id = idf$id
version = idf$version
print(version)

counts_file = file.path(version, "counts", paste0(id, ".csv.gz"))
if (!gcs_file_exists(counts_file)) {
  out = try({
    process_nhanes_80hz(id, version, verbose = TRUE)
  })
  if (!inherits(out, "try-error")) {
    gcs_upload_file(out$csv_file)
    gcs_upload_file(out$measures_file)
    gcs_upload_file(out$meta_file)
    gcs_upload_file(out$log_file)
    gcs_upload_file(out$counts_file)
  }
}
