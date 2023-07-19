#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(agcounts)
options(digits.secs = 3)
suppressPackageStartupMessages(library("optparse"))
source(here::here("code", "R", "helper_functions.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = Sys.getenv("SGE_TASK_ID")
ifold = as.numeric(ifold)
print(paste0("fold is: ", ifold))
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

df = df %>%
  dplyr::filter(file.exists(tarball_file))






max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(idf$tarball_file)

  files = list(
    tarball_file = idf$tarball_file,
    csv_file = idf$csv_file,
    log_file = idf$log_file,
    meta_file = idf$meta_file,
    counts_file = idf$counts_file,
    measures_file = idf$measures_file
  )

  if (!all(file.exists(unlist(files)))) {
    x = try({
      summarise_nhanes_80hz(
        csv_file = files$csv_file,
        log_file = files$log_file,
        meta_file = files$meta_file,
        counts_file = files$counts_file,
        measures_file = files$measures_file,
        sample_rate = 80L,
        dynamic_range = c(-6L, 6L),
        verbose = TRUE
      )})

    # doing this so .Last.value isn't maintained
    rm(x)
  }
}


