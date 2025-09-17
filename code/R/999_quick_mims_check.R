#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(agcounter)
library(SummarizedActigraphy)
library(readr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

df = df %>%
  dplyr::mutate(
    file_mims = here::here("data", "mims", version, paste0(id, ".csv.gz"))
  )

df = df %>%
  filter(file.exists(file_mims))

iid = 1
for (iid in seq(nrow(df))) {
  idf = df[iid,]
  print(paste0(iid, " of ", nrow(df)))

  print(idf$measures_file)
  measures = read_csv(idf$measures_file, progress = FALSE, show_col_types = FALSE)
  if ("MIMS_UNIT" %in% colnames(measures)) {
    measures = measures %>%
      select(time = HEADER_TIMESTAMP, MIMS_UNIT)
    mims = read_csv(idf$file_mims, progress = FALSE, show_col_types = FALSE) %>%
      select(time = HEADER_TIME_STAMP, MIMS_UNIT)
    stopifnot(isTRUE(all.equal(measures, mims)))
  } else {
    print("skipping")
  }

}
