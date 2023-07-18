library(magrittr)
library(dplyr)
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = Sys.getenv("SGE_TASK_ID")
ifold = as.numeric(ifold)
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

max_n = nrow(df)
for (index in seq(max_n)) {
  print(index)
  idf = df[index,]
  if (!file.exists(idf$file)) {
    dir.create(dirname(idf$file), showWarnings = FALSE,
               recursive = TRUE)
    print(idf$file)
    curl::curl_download(idf$url, idf$file)
  }
}
