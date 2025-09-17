#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(agcounter)
library(SummarizedActigraphy)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ids = c("73557", "73558", "73559", "73560", "73561",
          "80520", "80441", "80559", "74694", "74312", "74445", "74682",
        "80517", "74752", "74324")
df = df %>%
  filter(id %in% ids)

df = df %>%
  dplyr::mutate(
    file_mims = here::here("data", "mims", version, paste0(id, ".csv.gz"))
  )


# ifold = get_fold()
#
# if (!is.na(ifold)) {
#   df = df %>%
#     dplyr::filter(fold %in% ifold)
# }
force = FALSE

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  if (
    (!all(file.exists(idf$file_mims)) && file.exists(idf$csv_file)) ||
    force
  ) {
    data = read_80hz(idf$csv_file)
    data = data %>%
      rename(HEADER_TIME_STAMP = HEADER_TIMESTAMP)
    out = MIMSunit::mims_unit(
      data,
      epoch = "1 min",
      output_mims_per_axis = TRUE,
      dynamic_range = c(-6L, 6L)
    )
    write_csv_gz(out, idf$file_mims)
  }
}




