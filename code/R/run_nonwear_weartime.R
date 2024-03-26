# weartime::conda_create_weartime()
Sys.unsetenv("RETICULATE_PYTHON")
weartime::use_weartime_condaenv()
library(reticulate)
library(readr)
library(dplyr)
library(weartime)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

# set up SWAN


df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1
model_dir = here::here("weartime_models")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
model_path = file.path(model_dir, "cnn_v2_7.h5")
if (!file.exists(model_path)) {
  weartime::download_cnn_model(outdir = model_dir)
}

summarise_nonwear_second = function(df) {
  df %>%
    dplyr::mutate(
    time = lubridate::floor_date(time, "1 second")) %>%
    group_by(time) %>%
    summarise(
      wear = mean(wear, na.rm = TRUE) >= 0.5
    )
}

for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$csv_file
  print(file)
  dir.create(dirname(idf$nonwear_weartime_file), recursive = TRUE,
             showWarnings = FALSE)

  data = read_80hz(file)
  sample_rate = 80L
  nw_cnn = weartime::wt_cnn(df = data,
                            sample_rate = sample_rate,
                            model_path = model_path)
  nw_cnn_out = nw_cnn %>%
    summarise_nonwear_second() %>%
    dplyr::rename(HEADER_TIMESTAMP = time, wear_cnn = wear)

  nw_vmu = wt_vmu(df = data, sample_rate = sample_rate)
  nw_vmu_out = nw_vmu %>%
    summarise_nonwear_second() %>%
    dplyr::rename(HEADER_TIMESTAMP = time, wear_vmu = wear)

  # wt_xyz is wt_baseline
  nw_baseline = wt_baseline(df = data, sample_rate = sample_rate)
  nw_baseline_out = nw_baseline %>%
    summarise_nonwear_second() %>%
    dplyr::rename(HEADER_TIMESTAMP = time, wear_baseline = wear)

  nw_hees_2011 = wt_hees_2011(df = data, sample_rate = sample_rate)
  nw_hees_2011_out = nw_hees_2011 %>%
    summarise_nonwear_second() %>%
    dplyr::rename(HEADER_TIMESTAMP = time, wear_hees_2011 = wear)

  nw_hees_2013 = wt_hees_2013(df = data, sample_rate = sample_rate)
  nw_hees_2013_out = nw_hees_2013 %>%
    summarise_nonwear_second() %>%
    dplyr::rename(HEADER_TIMESTAMP = time, wear_hees_2013 = wear)

  nw_hees_optimized = wt_hees_optimized(df = data, sample_rate = sample_rate)
  nw_hees_optimized_out = nw_hees_optimized %>%
    summarise_nonwear_second() %>%
    dplyr::rename(HEADER_TIMESTAMP = time, wear_hees_optimized = wear)

  out = nw_cnn_out %>%
    full_join(nw_vmu_out, by = join_by(HEADER_TIMESTAMP)) %>%
    full_join(nw_baseline_out, by = join_by(HEADER_TIMESTAMP)) %>%
    full_join(nw_hees_2011_out, by = join_by(HEADER_TIMESTAMP)) %>%
    full_join(nw_hees_2013_out, by = join_by(HEADER_TIMESTAMP)) %>%
    full_join(nw_hees_optimized_out, by = join_by(HEADER_TIMESTAMP))

  write_csv_gz(out, idf$nonwear_weartime_file)
}
