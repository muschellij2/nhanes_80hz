library(magrittr)
library(dplyr)

index = 1

# process_index = function(index, cleanup = TRUE,
#                          upload = TRUE) {

stopifnot(length(index) == 1)
dfs = lapply(c("pax_h", "pax_g"), function(version) {
  readr::read_rds(
    here::here("data", "raw", paste0(version, "_filenames.rds"))
  )
})
df = dplyr::bind_rows(dfs)
xdf = df
# reorder them so we can download some random
# df = df[sample(nrow(df)),]
df = df %>%
  dplyr::filter(!file.exists(file))

# max_n = min(1000, nrow(df))
max_n = nrow(df)
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
