# swan::conda_create_swan()
Sys.unsetenv("RETICULATE_PYTHON")
swan::use_swan_condaenv()
library(reticulate)
library(readr)
library(dplyr)
library(swan)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

# set up SWAN


df = readRDS(here::here("data", "raw", "all_filenames.rds"))
# records that have only zeroes in their data
zero_df = readr::read_csv(here::here("data/raw/all_zero.csv.gz"))
df = left_join(df,
               zero_df %>%
                 select(id, version, all_zero) %>%
                 mutate(id = as.character(id)))

xdf = df

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1


for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf$csv_file
  print(file)
  dir.create(dirname(idf$nonwear_swan_file), recursive = TRUE,
             showWarnings = FALSE)

  if (!file.exists(idf$nonwear_swan_file) && !idf$all_zero) {
    data = read_80hz(file)
    nonwear = try({
      swan::swan(df = data, sampling_rate = 80L)
    })
    if (!inherits(nonwear, "try-error") && !is.null(nonwear)) {
      out = nonwear$first_pass %>%
        select(HEADER_TIMESTAMP = header_time_stamp,
               first_pass_predicted = predicted)
      sp = nonwear$second_pass %>%
        select(HEADER_TIMESTAMP = header_time_stamp,
               prediction,
               predicted,
               prob_sleep,
               prob_wear,
               prob_nonwear = prob_nwear,
               prob_sleep_smooth,
               prob_wear_smooth,
               prob_nonwear_smooth = prob_nwear_smooth
        )
      rm(nonwear)
      rm(data)
      sp$prob_sleep_smooth = sapply(sp$prob_sleep_smooth, identity)
      sp$prob_wear_smooth = sapply(sp$prob_wear_smooth, identity)
      sp$prob_nonwear_smooth = sapply(sp$prob_nonwear_smooth, identity)
      out = full_join(sp, out)
      out = out %>%
        mutate(wear = !prediction %in% "Nonwear",
               wear = ifelse(is.na(prediction) | prediction %in% "Unknown",
                             NA, wear)
        ) %>%
        select(HEADER_TIMESTAMP, wear, everything())
      stopifnot(!any(sapply(out, is.list)))
      write_csv_gz(out, idf$nonwear_swan_file)
    } else {
      rm(data)
    }
  }
}
