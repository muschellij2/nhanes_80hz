library(magrittr)
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = as.numeric(Sys.getenv("SGE_TASK_ID"))
if (is.na(ifold)) {
  ifold = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
}
print(paste0("fold is: ", ifold))
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

xdf = df

df = df %>%
  dplyr::filter(file.exists(tarball_file))

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(idf$tarball_file)
  files = list(
    tarball_file = idf$tarball_file,
    csv_file = idf$csv_file,
    log_file = idf$log_file,
    meta_file = idf$meta_file
  )

  if (!all(file.exists(unlist(files)))) {
    x = tarball_to_csv(
      tarball_file = files$tarball_file,
      csv_file = files$csv_file,
      log_file = files$log_file,
      meta_file = files$meta_file,
      num_threads = 1
    )
    # doing this so .Last.value isn't maintained
    rm(x)
  }
}
