library(magrittr)
source(here::here("code", "R", "helper_functions.R"))
version = "pax_h"

index = 1

# process_index = function(index, cleanup = TRUE,
#                          upload = TRUE) {

stopifnot(length(index) == 1)
xdf = df = readr::read_rds(
  here::here("data", "raw", paste0(version, "_filenames.rds"))
)


for (index in seq(nrow(df))) {
  print(index)
  idf = df[index,]
  if (!file.exists(idf$file)) {
    dir.create(dirname(idf$file), showWarnings = FALSE,
               recursive = TRUE)
    curl::curl_download(idf$url, idf$file)
  }
}
