library(magrittr)
library(dplyr)
source(here::here("code", "R", "helper_functions.R"))

index = 1

stopifnot(length(index) == 1)
dfs = lapply(c("pax_h", "pax_g"), function(version) {
  readr::read_rds(
    here::here("data", "raw", paste0(version, "_filenames.rds"))
  )
})
df = dplyr::bind_rows(dfs)
xdf = df

df = df %>%
  dplyr::filter(file.exists(file))

# raw = df$file[1]

max_n = min(1000, nrow(df))
for (index in seq(max_n)) {
  print(index)
  idf = df[index,]
  files = list(
    raw = idf$file,
    csv = idf$full_csv,
    logfile = idf$logfile,
    meta = idf$meta
  )

  if (!all(file.exists(unlist(files)))) {
    tarball_to_csv(raw = files$raw,
                   csv = files$csv,
                   logfile = files$logfile,
                   meta = files$meta
    )
  }
}
