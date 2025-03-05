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

  if (!file.exists(idf$nonwear_swan_file) && !idf$all_zero) {
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
    tudor_locke_sadeh = actigraph.sleepr::apply_tudor_locke(sadeh)
    tudor_locke_cole_kripke = actigraph.sleepr::apply_tudor_locke(cole_kripke)

    choi_df = purrr::map2_df(
      choi_nonwear$period_start, choi_nonwear$period_end,
      function(from, to) {
        data.frame(timestamp = seq(from, to, by = 60L),
                   choi_wear = FALSE)
      })
    data = left_join(data, choi_df) %>%
      tidyr::replace_na(list(choi_wear = TRUE))

  }
}
