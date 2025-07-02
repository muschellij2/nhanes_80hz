library(dplyr)
library(readr)
source("code/R/helper_functions.R")
logs = list.files(path = "data/logs/pax_h", pattern = ".csv", full.names = TRUE)
res = file.info(logs)
# res = res[res$size > 1000,]
logs = rownames(res)
df = lapply(logs, function(log_file) {
  id = sub("[.]csv.*", "", basename(log_file))
  log = read_log(log_file) %>%
    janitor::clean_names()
  log$id = id
  log = log %>%
    arrange(day_of_data, start_time)
})
df = bind_rows(df)

all_flags = sort(unique(df$data_quality_flag_code))

flag_types = c(
  "ADJACENT_INVALID",
  "CONTIGUOUS_ADJACENT_IDENTICAL_NON_ZERO_VALS_XYZ",
  "CONTIGUOUS_ADJACENT_ZERO_VALS_XYZ", "CONTIGUOUS_IMPOSSIBLE_GRAVITY",
  "CONTIGUOUS_MAX_G_VALS_X", "CONTIGUOUS_MAX_G_VALS_Y",
  "CONTIGUOUS_MAX_G_VALS_Z",
  "CONTIGUOUS_MIN_G_VALS_X", "CONTIGUOUS_MIN_G_VALS_Y",
  "CONTIGUOUS_MIN_G_VALS_Z",
  "COUNT_MAX_G_VALS_X", "COUNT_MAX_G_VALS_Y", "COUNT_MAX_G_VALS_Z",
  "COUNT_MIN_G_VALS_X", "COUNT_MIN_G_VALS_Y", "COUNT_MIN_G_VALS_Z",
  "COUNT_SPIKES_X", "COUNT_SPIKES_X_1S", "COUNT_SPIKES_Y",
  "COUNT_SPIKES_Y_1S",
  "COUNT_SPIKES_Z", "COUNT_SPIKES_Z_1S", "INTERVAL_JUMP_X",
  "INTERVAL_JUMP_Y",
  "INTERVAL_JUMP_Z")

stopifnot(all(flag_types %in% all_flags))

df = df %>%
  mutate(
    st = floor_date2(start_time, "1 minute"),
    et = floor_date2(end_time, "1 minute"),
  )
sdf = split(df, df$data_quality_flag_code)
# sdf = lapply(sdf, head)

floor_date2 = function(x, ...) {
  if (hms::is_hms(x)) {
    x = lubridate::hms(x)
  }
  x %>%
    lubridate::as_datetime() %>%
    lubridate::floor_date(...) %>%
    hms::as_hms()
}
all_minutes_same = sapply(sdf, function(x) {
  all(x$st == x$et)
})
