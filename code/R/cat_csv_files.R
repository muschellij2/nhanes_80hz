suppressPackageStartupMessages({
  library(dplyr)
})
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
df = readRDS(here::here("data", "raw", "all_filenames.rds"))
suppressWarnings({
  ifold = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
})

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

args <- commandArgs(trailingOnly = TRUE)
icol = "time_csv_file"
if (length(args) > 0) {
  icol = args[1]
}
cat(df[[icol]], sep = "\n")
