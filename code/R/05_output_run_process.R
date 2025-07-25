#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(agcounter)
library(SummarizedActigraphy)
options(digits.secs = 3)
suppressPackageStartupMessages(library("optparse"))
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}
force = TRUE

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  files = list(
    csv_file = idf$csv_file,
    log_file = idf$log_file,
    meta_file = idf$meta_file,
    counts_file = idf$counts_file,
    measures_file = idf$measures_file
  )
  # force re-run if MIMS_UNIT not found
  if (file.exists(idf$measures_file)) {
    x = readr::read_csv(idf$measures_file)
    if (!"MIMS_UNIT" %in% colnames(x)) {
      file.remove(idf$measures_file)
    }
  }

  if (
    (!all(file.exists(unlist(files))) && file.exists(idf$csv_file)) ||
    force
  ) {
    x = try({
      summarise_nhanes_80hz(
        csv_file = files$csv_file,
        log_file = files$log_file,
        meta_file = NULL,
        # meta_file = files$meta_file,
        # counts_file = files$counts_file,
        # make counts_file = NULL if don't want to run counts
        counts_file = NULL,
        calculate_mims = TRUE,
        measures_file = files$measures_file,
        sample_rate = 80L,
        dynamic_range = c(-6L, 6L),
        verbose = TRUE
      )})

    # doing this so .Last.value isn't maintained
    rm(x)
  }
}


