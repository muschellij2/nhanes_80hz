library(janitor)
library(tibble)
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

i = 1

df$all_zero = NA


for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$csv_file
  print(file)


  # data = read_80hz(file, progress = FALSE)
  # df$all_zero[i] = all(data$X == 0 & data$Y == 0 & data$Z == 0)
  data = data.table::fread(file, header = TRUE,
                           select = c("X", "Y", "Z"));
  df$all_zero[i] = data[, all(X == 0 & Y == 0 & Z ==0)]
}

out = df %>%
  select(id, version, all_zero, csv_file)
readr::write_csv(out, here::here("data/raw/all_zero.rds"))


