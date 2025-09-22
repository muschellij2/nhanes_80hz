#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(agcounter)
library(arrow)
library(readr)
library(SummarizedActigraphy)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

allow_truncation = FALSE
name_folder = ifelse(allow_truncation, "mims_truncation", "mims")
df = df %>%
  dplyr::mutate(
    file_mims = here::here("data", name_folder, version, paste0(id, ".csv.gz"))
  )

df = df %>%
  filter(file.exists(file_mims))

xpts = tibble(
  id = c("PAXMIN_Y", "PAXMIN_G", "PAXMIN_H"),
  file = here::here("data", "raw_min", paste0(id, ".XPT")),
  outfile = here::here("data", "raw_min", paste0(id, ".parquet"))
) %>%
  mutate(version = paste0("pax_", tolower(sub(".*_(.*)", "\\1", id))))

ss = split(xpts$outfile, xpts$version)

df$n_values = df$n_bad = NA

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # for (index in 36:nrow(df)) {
  # print(index)
  idf = df[index,]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  if (
    file.exists(idf$file_mims)
  ) {

    ds <- open_dataset(ss[[idf$version]])
    paxmims = ds %>%
      filter(SEQN %in% idf$id) %>%
      collect()

    mims = read_csv(idf$file_mims, progress = FALSE, show_col_types = FALSE)
    stopifnot(nrow(paxmims) == nrow(mims) |
                (nrow(paxmims) + 1) == nrow(mims))
    mims = mims[1:nrow(paxmims),]

    mims = mims %>%
      select(time = HEADER_TIME_STAMP, MIMS_UNIT)

    paxmims$MIMS_UNIT = mims$MIMS_UNIT

    paxmims = paxmims %>%
      mutate(diff = abs(MIMS_UNIT - PAXMTSM),
             pct_diff = diff / ((MIMS_UNIT + PAXMTSM)/2) * 100) %>%
      select(PAXDAYM, PAXSSNMP, PAXTSM, PAXMTSM, MIMS_UNIT, diff,
             pct_diff)

    head(paxmims)
    good = paxmims$pct_diff < 1 | paxmims$diff < 0.1
    df$n_values[index] = nrow(paxmims)
    df$n_bad[index] = sum(!good)
    print(sum(!good))
  }
}

out = df %>%
  select(id, version, n_values, n_bad)
fname = here::here("data", "mims_comparison_check.rds")
readr::write_rds(out, fname)




