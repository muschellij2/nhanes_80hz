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
  print(paste0("File ", index, "of ", max_n, ": ", idf$csv_file))
  files = list(
    counts_1s_file = idf$counts_1s_file
  )
  dir.create(dirname(idf$counts_1s_file), showWarnings = FALSE, recursive = TRUE)

  if (!all(file.exists(unlist(files)))) {
    x = agcounter::convert_counts_csv(
      file = idf$csv_file,
      outfile = idf$counts_1s_file,
      sample_rate = 80L,
      epoch_in_seconds = 1L,
      time_column = "HEADER_TIMESTAMP"
    )
    # df = read_80hz(idf$csv_file)
    # x = agcounter::get_counts(
    #   df,
    #   sample_rate = 80L,
    #   epoch_in_seconds = 1L
    # )
    # doing this so .Last.value isn't maintained
    rm(x)
  }
}
