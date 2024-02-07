library(googleCloudRunner)
print('gr')
library(googleCloudStorageR)
print('gcs')
library(trailrun)
print('trailrun')
library(dplyr)
print("dplyr")
library(tidyr)
print('tidyr')
library(readr)
library(vroom)
print('vroom')
library(bigrquery)
print('bq')
library(streamliner)
print('streamliner')
source("helper_functions.R")
print("helper added")

version = Sys.getenv("VERSION", unset = NA)
if (is.na(version)) {
  version = "pax_h"
}
# THIS IS NOT CORRECT CODE - shit.

options(digits.secs = 3)
print("setting up bucket")
bucket = "nhanes_80hz"
bucket_setup(bucket)
print("gce setup")
config = cr_gce_setup(bucket = bucket)
print("bucket setup")

if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}
wide = wide[wide$version %in% version,]

bqt = bq_table(project = config$project,
               dataset = bucket,
               table = paste0(version, "_meta"))
meta_df = tbl(bqt) %>%
  collect() %>%
  mutate(id = as.character(id))
wide = left_join(wide, meta_df)
# real bad 76180, long is 75319, 74484 is bad
# hmm 78159
bad_ids = c(74484, 76180, 78159)
# real bad 65416
bad_ids = c(bad_ids, 65416)
bad_ids = c(bad_ids, readLines("bad_ids.txt"))

ifile = which(wide$id== 78933)
for (ifile in seq(nrow(wide))) {
  idf = wide[ifile,]
  csv_file = idf$csv
  log_file = idf$logs
  flag_file = idf$flags
  print(idf$id)
  if (idf$id %in% bad_ids) {
    next
  }

  if (is.null(flag_file) || is.na(flag_file)) {

    flag_file = sub("logs", "flags", log_file)
    dir.create(dirname(flag_file), showWarnings = FALSE, recursive = TRUE)

    if (!file.exists(csv_file)) {
      gcs_download(csv_file)
      all_minutes = read_80hz(csv_file)
      file.remove(csv_file)
    } else {
      all_minutes = read_80hz(csv_file)
    }
    all_minutes = all_minutes$HEADER_TIMESTAMP
    all_minutes = range(all_minutes)
    all_minutes = lubridate::floor_date(all_minutes, "1 minute")
    all_minutes = seq(all_minutes[1], all_minutes[2], by = 60L)
    all_minutes = unique(all_minutes)

    if (!file.exists(log_file)) {
      gcs_download(log_file, overwrite = TRUE)
    }
    log = read_log(log_file) %>%
      janitor::clean_names()
    log = log %>%
      arrange(day_of_data, start_time)

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
    ################################################
    # See if flag exists
    ################################################
    # if (!file.exists(csv_file)) {
    #   gcs_download(csv_file)
    # }
    # df = read_80hz(csv_file)
    # file.remove(csv_file)
    # df = df %>%
    #   dplyr::select(HEADER_TIMESTAMP)
    # df = df %>%
    #   dplyr::mutate(HEADER_TIMESTAMP =
    # SummarizedActigraphy:::floor_sec(HEADER_TIMESTAMP)) %>%
    #   dplyr::distinct()
    df = tibble::tibble(
      HEADER_TIMESTAMP =
        seq(SummarizedActigraphy:::floor_sec(idf$start_time),
            SummarizedActigraphy:::floor_sec(idf$stop_time),
            by = "1 sec"))
    min_day = min(df$HEADER_TIMESTAMP)
    min_day = lubridate::floor_date(min_day, unit = "days")
    min_day = lubridate::as_date(min_day)
    rdf = range(df$HEADER_TIMESTAMP)
    flags = matrix(0, nrow = nrow(df), ncol = length(flag_types))
    colnames(flags) = flag_types
    flags = tibble::as_tibble(flags)
    flags = dplyr::bind_cols(df, flags)


    if (nrow(log) > 0) {

      log = log %>%
        mutate(date = min_day + day_of_data - 1,
               start_dt = paste0(as.character(date), " ", as.character(start_time)),
               end_dt = paste0(as.character(date), " ", as.character(end_time))
        )
      log = log %>%
        dplyr::mutate(start_dt = lubridate::ymd_hms(start_dt),
                      end_dt = lubridate::ymd_hms(end_dt))
      add_day0 = lubridate::hour(log$start_time) >= 20 &
        lubridate::hour(log$end_time) == 0
      add_day = lubridate::hour(log$start_time) >= 20 &
        lubridate::hour(log$end_time) < 20
      if (any(add_day & !add_day0)) {
        message(paste0("Weird dates for id ", idf$id))
      }
      log$end_dt[add_day] = log$end_dt[add_day] + lubridate::as.period(1, "day")

      good = SummarizedActigraphy:::floor_sec(log$start_dt) >= rdf[1] &
        SummarizedActigraphy:::floor_sec(log$end_dt) <= rdf[2]
      if (!all(good)) {
        next
      }
      stopifnot(all(good))

      log$start_dt = SummarizedActigraphy:::floor_sec(log$start_dt)
      log$end_dt = SummarizedActigraphy:::floor_sec(log$end_dt)
      i = 1
      for (i in seq(nrow(log))) {
        flag_code =log$data_quality_flag_code[i]
        start_dt = log$start_dt[i]
        end_dt = log$end_dt[i]
        times = seq(start_dt, end_dt, by = 1)
        rows = flags$HEADER_TIMESTAMP %in% times
        flags[rows, flag_code] = flags[rows, flag_code] +
          rep(log$data_quality_flag_value[i], length = length(rows))
        if (i %% 1000 == 0) print(i)
      }
    }
    readr::write_csv(flags, flag_file)
    gcs_upload_file(flag_file)
  }
}
# make_flag_df = function(csv, log) {

# }
