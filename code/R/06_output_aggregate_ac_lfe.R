# translates NHANES PAXMIN files into same files but with time stamps instead of samples

library(dplyr)
library(readr)
library(lubridate)
library(tidyr)
options(digits.secs = 3)
source(here::here("code/R/utils.R"))
source(here::here("code/R/helper_functions.R"))

fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
df = df %>%
  mutate(rawmin_file = here::here("data/raw_min", df$version, paste0(df$id, ".csv.gz")))
xdf = df

# waves = c("G", "H", "Y")
# hd_files = here::here("data", "raw", paste0("PAXHD_", waves, ".XPT"))
# names(hd_files) = paste("pax_", tolower(waves))
# hd = purrr::map_df(hd_files, haven::read_xpt, .id = "version")


max_n = nrow(df)
index = 1
all_data = vector(mode = "list", length = max_n)
for (index in seq(max_n)) {
  idf = df[index, ]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  min_file = file.path(here::here("data", "min", idf$version, paste0(idf$id, ".csv.gz")))

  x =  readr::read_csv(min_file,
                       col_types = cols(PAXFLGSM = col_character(),
                                        SEQN = col_character(),
                                        PAXDAYM = col_integer(),
                                        PAXDAYWM = col_integer(),
                                        PAXPREDM = col_integer()),
                       progress = FALSE)
  readr::stop_for_problems(x)
  x = x %>%
    dplyr::select(SEQN, PAXDAYM, PAXDAYWM, time) %>%
    mutate(date = floor_date(time, unit = "day"),
           date = as_date(date))
  x = x %>%
    distinct(SEQN, PAXDAYM, PAXDAYWM, date)

  counts = readr::read_csv(idf$counts_60s_lfe_from_30Hz_file,
                           progress = FALSE, show_col_types = FALSE,
                           col_types =
                             cols(
                               HEADER_TIMESTAMP = col_datetime(format = ""),
                               X = col_double(),
                               Y = col_double(),
                               Z = col_double(),
                               AC = col_double()
                             ))
  readr::stop_for_problems(counts)

  counts = counts %>%
    mutate(SEQN = as.character(idf$id)) %>%
    select(SEQN, time = HEADER_TIMESTAMP, AC)
  counts = counts %>%
    mutate(
      min = as.numeric(difftime(hms::as_hms(time),
                                hms::as_hms("00:00:00"),
                                units = "mins")) + 1,
      min = sprintf("min_%04d", min),
      PAXDAYM_counts = as.integer(
        difftime(floor_date(time, "day"),
                 floor_date(time, "day")[1],
                 units = "days")) + 1,
      date = as_date(floor_date(time, unit = "1 day"))
    )
  counts = counts %>%
    left_join(x, by = join_by(SEQN, date))
  stopifnot(all(counts$PAXDAYM_counts == counts$PAXDAYM))
  counts = counts %>%
    select(-any_of("PAXDAYM_counts"))
  data_subset = counts %>%
    select(SEQN, contains("PAXDAY"), min, AC) %>%
    tidyr::pivot_wider(
      id_cols = c(SEQN, PAXDAYM, PAXDAYWM),
      names_from = min,
      values_from = AC,
      names_sort = TRUE
    ) %>%
    mutate(across(c(SEQN, PAXDAYM, PAXDAYWM), ~ as.integer(.x)))
  data_subset$version = idf$version
  all_data[[index]] = data_subset
}

data = dplyr::bind_rows(all_data)

outfile = here::here("data/counts_60s_lfe_from30Hz",
                     "combined_wide_counts_lfe.rds")
readr::write_rds(data, outfile, compress = "xz")

ss = split(data, data$version)
for (version in c("pax_y", "pax_g", "pax_h")) {
  dfi = ss[[version]]
  outfile = here::here("data/counts_60s_lfe_from30Hz",
                       paste0("combined_wide_counts_lfe_", version, ".rds"))
  readr::write_rds(dfi, outfile, compress = "xz")
}



