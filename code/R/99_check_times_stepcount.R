library(janitor)
library(tibble)
library(stepcount)
library(dplyr)
library(readr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df
zero_df = readr::read_csv(here::here("data/raw/all_zero.csv.gz"))
df = left_join(df,
               zero_df %>%
                 select(id, version, all_zero) %>%
                 mutate(id = as.character(id)))


i=1

df$stepcount_stop_time = df$stepcount_start_time = lubridate::NA_POSIXct_
for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  print(idf$stepcount_file)
  if (file.exists(idf$stepcount_file)) {
    data = readr::read_csv(
      idf$stepcount_file,
      col_types = cols(
        time = col_datetime(format = ""),
        steps = col_double(),
        walking = col_logical(),
        non_wear = col_logical()
      ),
      progress = FALSE
    )
    r = range(data$time)
    df$stepcount_start_time[i] = r[1]
    df$stepcount_stop_time[i] = r[2]
  }
}

df$rf_stepcount_stop_time = df$rf_stepcount_start_time = lubridate::NA_POSIXct_
for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  print(idf$rf_stepcount_file)
  if (file.exists(idf$rf_stepcount_file)) {
    data = readr::read_csv(
      idf$rf_stepcount_file,
      col_types = readr::cols(
        time = col_datetime(format = ""),
        steps = col_double(),
        walking = col_logical(),
        non_wear = col_logical()
      ),
      progress = FALSE
    )
    r = range(data$time)
    df$rf_stepcount_start_time[i] = r[1]
    df$rf_stepcount_stop_time[i] = r[2]
  }
}

df = df %>%
  select(id, version, csv_file,
         stepcount_start_time, stepcount_stop_time,
         rf_stepcount_start_time, rf_stepcount_stop_time,
  )
write_csv_gz(df, here::here("data/checks/check_times_stepcount.csv.gz"))
