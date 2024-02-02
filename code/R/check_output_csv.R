library(magrittr)
library(dplyr)
library(walking)

options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = get_fold()
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

iid = 1
for (iid in seq_len(nrow(df))) {
  idf = df[iid,]
  message(paste0("Reading in ", idf$csv_file))
  # data = read_80hz(idf$csv_file, progress = FALSE, col_select = HEADER_TIMESTAMP)
  data = data.table::fread(idf$csv_file, select = "HEADER_TIMESTAMP")
  check_time_diffs(data$HEADER_TIMESTAMP, sample_rate = 80L)
  rm(data)
}

