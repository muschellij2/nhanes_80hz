library(janitor)
library(tibble)
library(readr)
reticulate::use_condaenv("asleep")
library(asleep)
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

model_path = here::here("asleep_models", "ssl.joblib.lzma")

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df
zero_df = readr::read_csv(here::here("data/raw/all_zero.csv.gz"))
df = left_join(df,
               zero_df %>%
                 select(id, version, all_zero) %>%
                 mutate(id = as.character(id)))

ifold = get_fold()

if (!all(is.na(ifold))) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1
sleep_cols = c("sleep_file", "sleep_output_file")

for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$time_csv_file
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  print(file)
  outfiles = unlist(idf[,sleep_cols])
  if (!all(file.exists(outfiles)) && !idf$all_zero) {

    out = try({
      asleep(file = idf$time_csv_file,
             model_path = model_path)
    })
    # errors can happen if all the data is zero
    if (!inherits(out, "try-error")) {
      try({
        lapply(out$paths, file.remove)
      })
      readr::write_rds(out, idf$sleep_output_file, compress = "gz")
      write_csv_gz(out$predictions, idf$sleep_file)
    }
    rm(out)
  }
}
