library(janitor)
library(tibble)
library(readr)
reticulate::use_condaenv("actinet")
library(actinet)
library(dplyr)
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

ifold = get_fold()

if (!all(is.na(ifold))) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1
sleep_cols = c("actinet_file")

model_dir = here::here("actinet_models")

out = try({
  actinet::actinet(file = df$time_csv_file[1], cache_classifier = TRUE)
})
for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$time_csv_file
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  print(file)
  outfiles = unlist(idf[,sleep_cols])
  if (!all(file.exists(outfiles)) && !idf$all_zero) {
    out = try({
      actinet::actinet(file = idf$time_csv_file, sample_rate = 80L,
                       classifier = "walmsley")
    })
    # errors can happen if all the data is zero
    if (!inherits(out, "try-error")) {
      file.copy(out$outfiles[1], idf$actinet_file, overwrite = TRUE)
    }
    rm(out)
  }
}
