library(magrittr)
library(dplyr)
source(here::here("code", "R", "helper_functions.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = Sys.getenv("SGE_TASK_ID")
ifold = as.numeric(ifold)
print(paste0("fold is: ", ifold))
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

xdf = df

df = df %>%
  dplyr::filter(file.exists(file))

max_n = nrow(df)
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
                       meta = files$meta,
                       num_threads = 1
    )
    # doing this so .Last.value isn't maintained
    rm(x)
  }
}
