# Run nonwear detection using Choi's algorithm
library(actigraph.sleepr)
library(readr)
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))


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
  file = idf$counts_60s_file
  print(file)

  if (!all(file.exists(c(
    idf$nonwear_ac60_file,
    idf$sleep_ac60_file))) && !idf$all_zero) {
    data = read_csv(file)
    data = data %>%
      rename(timestamp = HEADER_TIMESTAMP,
             axis1 = X,
             axis2 = Y,
             axis3 = Z)
    mode(data$timestamp) = "double"
    stopifnot(!actigraph.sleepr::has_missing_epochs(data))
    choi_nonwear = actigraph.sleepr::apply_choi(data, use_magnitude = TRUE)
    troiano_nonwear = actigraph.sleepr::apply_troiano(data, use_magnitude = TRUE)
    cole_kripke = actigraph.sleepr::apply_cole_kripke(data)
    sadeh = actigraph.sleepr::apply_sadeh(data)
    # tudor_locke_sadeh = actigraph.sleepr::apply_tudor_locke(sadeh)
    # tudor_locke_cole_kripke = actigraph.sleepr::apply_tudor_locke(cole_kripke)

    choi_df = convert_period_to_data(choi_nonwear, data = data)
    choi_df = choi_df %>%
      select(timestamp, choi_wear = wear)
    troiano_df = convert_period_to_data(troiano_nonwear, data = data)
    troiano_df = troiano_df %>%
      select(timestamp, troiano_wear = wear)
    sadeh_df = sadeh %>%
      select(timestamp, sadeh_sleep = sleep)
    ck_df = cole_kripke %>%
      select(timestamp, cole_kripke_sleep = sleep)

    res = full_join(choi_df, troiano_df)
    res = res %>%
      rename(HEADER_TIMESTAMP = timestamp)
    stopifnot(nrow(res) == nrow(data))
    write_csv_gz(res, idf$nonwear_ac60_file)

    res = full_join(sadeh_df, ck_df)
    res = res %>%
      rename(HEADER_TIMESTAMP = timestamp)
    stopifnot(nrow(res) == nrow(data))
    write_csv_gz(res, idf$sleep_ac60_file)
  }
}
