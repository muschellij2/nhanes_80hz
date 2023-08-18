library(magrittr)
library(dplyr)
library(walking)

options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))

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


df = df %>%
  dplyr::select(id, fold, version, starts_with("csv")) %>%
  dplyr::rename(data_file = csv_file) %>%
  dplyr::select(-dplyr::any_of(c("csv_file", "csv80_file")))
df = df %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("csv"),
                      names_to = "sample_rate",
                      values_to = "file") %>%
  dplyr::mutate(
    sample_rate = sub("csv", "", sample_rate),
    sample_rate = sub("_.*", "", sample_rate)) %>%
  dplyr::arrange(sample_rate, fold, id)

sdf = split(df, df$id)
uids = unique(df$id)
iid = uids[1]
for (iid in uids) {
  idf = sdf[[iid]]
  idf$fe = file.exists(idf$file)
  csv_file = idf$data_file[1]
  if (!all(idf$fe)) {
    idf = idf %>%
      dplyr::filter(!fe)
    # acc_data = readr::read_csv(csv_file, num_threads = 1, guess_max = Inf)
    acc_data = read_80hz(csv_file)
    message(paste0("Checking for problems in ", csv_file))
    readr::stop_for_problems(acc_data)
    irow = 1
    for (irow in seq(nrow(idf))) {
      new_sample_rate = idf$sample_rate[irow] %>% as.integer()
      new_data = walking::resample_accel_data(
        data = acc_data,
        sample_rate = new_sample_rate)
      print("Writing Data")
      write_csv_gz(new_data, file = idf$file[irow])
      print(idf$file[irow])
      rm(new_data)
      gc()
    }
    rm(acc_data)
    gc()
  }
}
