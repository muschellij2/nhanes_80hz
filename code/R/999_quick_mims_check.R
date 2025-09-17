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
    file_mims = here::here("data", "mims", version, paste0(id, ".csv.gz")),
    file_mims_raw = here::here("data", "raw_min", version, paste0(id, ".csv.gz"))
  )

df = df %>%
  filter(file.exists(file_mims))

iid = 1
for (iid in seq(nrow(df))) {
  idf = df[iid,]
  print(paste0(iid, " of ", nrow(df)))

  print(idf$measures_file)
  measures = read_csv(idf$measures_file, progress = FALSE, show_col_types = FALSE)
  raw_min = read_csv(idf$file_mims_raw)
  stopifnot(nrow(measures) == nrow(raw_min))
  if ("MIMS_UNIT" %in% colnames(measures)) {
    measures = measures %>%
      select(time = HEADER_TIMESTAMP, MIMS_UNIT)
    raw_min$time = measures$time
    raw_min = raw_min %>%
      select(time, MIMS_UNIT = PAXMTSM)
    mims = read_csv(idf$file_mims, progress = FALSE, show_col_types = FALSE) %>%
      select(time = HEADER_TIME_STAMP, MIMS_UNIT)
    stopifnot(isTRUE(all.equal(measures, mims)))
    raw_min = raw_min %>%
      mutate(MIMS_UNIT = round(MIMS_UNIT, 2))
    mims = mims %>%
      mutate(MIMS_UNIT = round(MIMS_UNIT, 2))
    stopifnot(isTRUE(all.equal(raw_min, mims)))
  } else {
    print("skipping")
  }

}
