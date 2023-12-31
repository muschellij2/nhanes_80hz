library(magrittr)
library(dplyr)
library(write.gt3x)
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

# x = normalizePath(
#   list.files(path = "data/acc_csv",
#              full.names = TRUE,
#              recursive = TRUE,
#              pattern = ".csv")
# )

df = df %>%
  dplyr::filter(file.exists(csv_file))

max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(paste0("File ", index, " of ", max_n, ": ", idf$csv_file))
  files = list(
    acc_csv_file = idf$acc_csv_file
  )
  dir.create(dirname(idf$acc_csv_file), showWarnings = FALSE, recursive = TRUE)

  if (!all(file.exists(unlist(files)))) {
    if (!file.exists(idf$acc_csv_file)) {
      data = read_80hz(idf$csv_file, progress = FALSE)
      data = data %>%
        dplyr::rename(time = any_of(c("HEADER_TIMESTAMP", "HEADER_TIME_STAMP")))
      write.gt3x::write_actigraph_csv(
        df = data,
        file = idf$acc_csv_file,
        sample_rate = 80L,
        max_g = "8",
        progress = FALSE)
      # doing this so .Last.value isn't maintained
      rm(data)
    }

  }
}
