library(magrittr)
library(dplyr)
library(write.gt3x)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
x = readr::read_csv(here::here("data/test_acc_csv/substudy/substudy_ids.csv")) %>%
  mutate(id = as.character(id))
xdf = df

df = df %>%
  right_join(x)

df = df %>%
  select(id, csv10_file, csv15_file)

df = df %>%
  tidyr::gather(value = "csv_file", key = "sample_rate", starts_with("csv")) %>%
  mutate(sample_rate = sub(".*csv(\\d*)_.*", "\\1", sample_rate))


df$csv_file = sub("pax_(h|g|y)/", "",
                    sub("data/", "data/test_acc_csv/substudy/",
                        df$csv_file))
df$acc_csv_file = sub("/csv", "/acc_csv", df$csv_file)


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
      data = read_80hz(idf$csv_file)
      data = data %>%
        dplyr::rename(time = any_of(c("HEADER_TIMESTAMP", "HEADER_TIME_STAMP")))
      write.gt3x::write_actigraph_csv(
        df = data,
        file = idf$acc_csv_file,
        sample_rate = as.integer(idf$sample_rate),
        max_g = "8")
      # doing this so .Last.value isn't maintained
      rm(data)
    }
  }
}
