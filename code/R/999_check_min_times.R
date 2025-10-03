# translates NHANES PAXMIN files into same files but with time stamps instead of samples

library(dplyr)
library(readr)
library(lubridate)
library(nhanesA)
options(digits.secs = 3)
source(here::here("code/R/utils.R"))
source(here::here("code/R/helper_functions.R"))

fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
df = df %>%
  mutate(rawmin_file = here::here("data/raw_min", idf$version, paste0(idf$id, ".csv.gz")))
xdf = df
ifold = get_fold()
if(!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

force = FALSE

waves = c("G", "H", "Y")
hd_files = here::here("data", "raw", paste0("PAXHD_", waves, ".XPT"))
names(hd_files) = paste("pax_", tolower(waves))
hd = purrr::map_df(hd_files, haven::read_xpt, .id = "version")


max_n = nrow(df)
index = 1
for (index in seq(max_n)) {
  idf = df[index, ]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  meta_df = read_csv(idf$meta_file, n_max = 1, progress = FALSE,
                     col_types = cols(
    date = col_datetime(format = ""),
    .default = col_character()
  ))
  meta_df = meta_df %>%
    distinct(date)

  start_time = floor_date(min(meta_df$date), "1 min")

  cmd = paste0("pigz -dc ", idf$csv_file,  "| (sed -n '2p;$p') | awk -F ',' '{print $1}'")
  output = system(cmd, intern = TRUE)

  ihd = hd %>%
    filter(SEQN %in% idf$id) %>%
    select(PAXFTIME, PAXETLDY)

  start_time_raw = as.POSIXct(output[1], format = "%Y-%m-%dT%H:%M:%OSZ", tz="UTC")
  end_time_raw = as.POSIXct(output[2], format = "%Y-%m-%dT%H:%M:%OSZ", tz="UTC")

  stopifnot(start_time == start_time_raw)
  stopifnot(ihd$PAXFTIME == as.character(hms::as_hms(start_time)))
  end_times = as.character(hms::as_hms(c(floor_date(end_time_raw, "1 sec"),
                                         round_date(end_time_raw, "1 sec"),
                                         ceiling_date(end_time_raw, "1 sec"))))
  stopifnot(ihd$PAXETLDY %in% end_times )


}
