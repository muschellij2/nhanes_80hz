library(magrittr)
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

xdf = df

df = df %>%
  dplyr::filter(file.exists(csv_file))

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(idf$csv_file)
  outfile = idf$time_csv_file
  if (!all(file.exists(outfile))) {
    file = rename_xyzt(csv_file = idf$csv_file)
    R.utils::gzip(
      file,
      destname = outfile,
      overwrite = TRUE,
      remove = TRUE)
    suppressWarnings({
      file.remove(file)
    })
  }
}
