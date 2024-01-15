library(janitor)
library(tibble)
library(stepcount)
unset_reticulate_python()
use_stepcount_condaenv()
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

model_path = here::here("stepcount_models/ssl-20230208.joblib.lzma")
model_type = "ssl"
if (!file.exists(model_path)) {
  model_path = NULL
}
sample_rate = 30L
stepcount_col = ifelse(sample_rate != 80L, "stepcount_file",
                paste0("stepcount", sample_rate, "_file"))
csv_col = ifelse(sample_rate != 80L, "csv_file",
                       paste0("csv", sample_rate, "_file"))

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1
for (i in seq_len(nrow(df))) {
  idf = df[i,]
  file = idf[[csv_col]]
  data = read_80hz(file, progress = FALSE)

  out = stepcount(data, model_path = model_path, model_type = model_type)
  rm(list = "data")
  info = tibble::as_tibble(out$info)
  info = janitor::clean_names(info)
  info$filename = file

  # nonwear = out$processed_data
  # nonwear = reticulate::py_to_r(nonwear)
  # times = attr(nonwear, "pandas.index")
  # times = reticulate::py_to_r(times$values)
  # nonwear$time = times
  # nonwear = data %>%
  #   dplyr::mutate(
  #     na_x = is.na(x),
  #     time = lubridate::floor_date(time, "10 second")) %>%
  #   dplyr::group_by(time) %>%
  #   dplyr::summarise(
  #     non_wear = any(na_x)
  #   )
  result = dplyr::full_join(out$steps, out$walking)
  result = result %>%
    dplyr::mutate(non_wear = is.na(steps) & is.na(walking))
  readr::write_csv(result, idf[[stepcount_col]])
}
