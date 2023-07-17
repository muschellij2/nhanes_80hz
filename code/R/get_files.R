library(magrittr)
library(dplyr)
source(here::here("code", "R", "helper_functions.R"))
version = "pax_h"

index = 1

# process_index = function(index, cleanup = TRUE,
#                          upload = TRUE) {

stopifnot(length(index) == 1)
xdf = df = readr::read_rds(
  here::here("data", "raw", paste0(version, "_filenames.rds"))
)
df = df %>%
  dplyr::filter(!file.exists(file))

max_n = min(1000, nrow(df))
# for (index in seq(nrow(df))) {
for (index in seq(max_n)) {
  print(index)
  idf = df[index,]
  if (!file.exists(idf$file)) {
    dir.create(dirname(idf$file), showWarnings = FALSE,
               recursive = TRUE)
    curl::curl_download(idf$url, idf$file)
  }
}
