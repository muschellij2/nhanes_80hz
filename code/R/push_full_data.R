library(googleCloudRunner)
library(bigrquery)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
library(tidyr)
library(streamliner)
options(digits.secs = 3)
source("helper_functions.R")
version = "pax_h"
bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)
config = trailrun::cr_gce_setup(bucket = bucket)

##########################################
# Read in the wide data
##########################################
if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}

data = wide[ wide$version %in% version, ]
table = paste0(version, "_meta")
bqt = bq_table(project = config$project,
               dataset = bucket,
               table = table)



data_bqt = bq_table(project = config$project,
                    dataset = bucket,
                    table = paste0(version, "_data"))

measure_1min_bqt = bq_table(project = config$project,
                            dataset = bucket,
                            table = paste0(version, "_measures_1min"))
in_bqt = function(bqt, id) {
  if (bigrquery::bq_table_exists(bqt)) {
    xx = tbl(bqt)
    xx = xx %>%
      count(id) %>%
      collect()
    return(id %in% xx$id)
  } else {
    return(FALSE)
  }
}

meta_df = tbl(bqt) %>%
  collect()
wide = wide %>%
  filter(id %in% meta_df$id)

index = 1
run_data = wide[index,]
run_id = id = run_data$id


has_been_run = in_bqt(data_bqt)

if (!has_been_run) {

  meta_idf = meta_df %>%
    filter(id %in% run_data$id)
  start_time = meta_idf$start_time
  stop_time = meta_idf$stop_time
  rm(meta_idf)

  gz_file = run_data$csv
  if (!file.exists(gz_file)) {
    gcs_download(gz_file)
  }

  csv_file = R.utils::gunzip(
    gz_file, temporary = TRUE, remove = FALSE,
    overwrite = TRUE)
  file.remove(gz_file)

  col_types = vroom::cols(
    # HEADER_TIMESTAMP = col_datetime_with_frac_secs(),
    HEADER_TIMESTAMP = vroom::col_datetime(),
    X = vroom::col_double(),
    Y = vroom::col_double(),
    Z = vroom::col_double()
  )
  message("Reading in Full data")
  dat = vroom::vroom(csv_file, col_types = col_types)
  file.remove(csv_file)
  frac_seconds = as.numeric(dat$HEADER_TIMESTAMP) %% 1
  stopifnot(any(frac_seconds > 0))
  dat = dat %>%
    rename(HEADER_TIME_STAMP = HEADER_TIMESTAMP)

  check_csv = function(dat, start_time, stop_time) {
    # dat = vroom::vroom(csv_file)
    r = range(dat$HEADER_TIME_STAMP)
    assertthat::assert_that(
      r[1] == start_time,
      r[2] == stop_time)
    dtime = as.numeric(diff(dat$HEADER_TIME_STAMP, units = "secs"))
    eps = 0.000001
    assertthat::assert_that(
      all(dtime > 0),
      mean(dtime) <= (1/80 + eps)
    )
    rm(dtime)
    rm(dat)
    gc()
    gc()
    return(TRUE)
  }
  check_csv(dat, start_time, stop_time)

  # Add to database!!!
  stopifnot(
    lubridate::is.POSIXct(dat$HEADER_TIME_STAMP)
  )
  message("Uploading Full data")

  dat$id = run_id
  dat = dat %>%
    select(id, everything()) %>%
    as.data.frame()
  push_table_up(data_bqt, dat)
  rm(dat)
}
