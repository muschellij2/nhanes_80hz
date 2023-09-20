library(magrittr)
library(dplyr)
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = get_fold()
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  print(index)
  idf = df[index,]
  if (!file.exists(idf$tarball_file)) {
    dir.create(dirname(idf$tarball_file), showWarnings = FALSE,
               recursive = TRUE)
    print(idf$tarball_file)
    curl::curl_download(idf$url, idf$tarball_file)
  }
}
