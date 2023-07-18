library(magrittr)
library(dplyr)
source(here::here("code", "R", "helper_functions.R"))
n_folds = 218
index = 1

stopifnot(length(index) == 1)
dfs = lapply(c("pax_h", "pax_g"), function(version) {
  readr::read_rds(
    here::here("data", "raw", paste0(version, "_filenames.rds"))
  )
})
df = dplyr::bind_rows(dfs)
df = df %>%
  mutate(fold = seq(dplyr::n()),
         fold = floor(fold / floor(dplyr::n()/n_folds) + 1))
ifold = Sys.getenv("SGE_TASK_ID")
ifold = as.numeric(ifold)
if (is.na(ifold)) {
  ifold = 2
}

print(ifold)
xdf = df

df = df %>%
  dplyr::filter(fold %in% ifold)
df = df %>%
  dplyr::filter(file.exists(file))

# raw = df$file[1]

max_n = min(1000, nrow(df))
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(idf$file)
  files = list(
    raw = idf$file,
    csv = idf$full_csv,
    logfile = idf$logfile,
    meta = idf$meta
  )

  if (!all(file.exists(unlist(files)))) {
    x = tarball_to_csv(raw = files$raw,
                       csv = files$csv,
                       logfile = files$logfile,
                       meta = files$meta
    )
    # doing this so .Last.value isn't maintained
    rm(x)
  }
}
