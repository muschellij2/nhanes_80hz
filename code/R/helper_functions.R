options(digits.secs = 3)
options(readr.num_threads = 1)

have_pigz = function() {
  pigz = Sys.which("pigz")
  !nzchar(pigz) || nchar(system("which pigz", intern = TRUE)) > 0
}

col_types_80hz = vroom::cols(
  # HEADER_TIMESTAMP = col_datetime_with_frac_secs(),
  HEADER_TIMESTAMP = vroom::col_datetime(),
  X = vroom::col_double(),
  Y = vroom::col_double(),
  Z = vroom::col_double()
)

download_80hz = function(id, version, exdir = tempdir(check = TRUE),
                         quiet = FALSE, ...) {
  files = id
  # it should be able to pull from the current URL and get the
  # necessary bytes for each file in the pub/{version}
  # Cache any downloading of the URL/HTML so that you're not crushing
  # the site with repeated use
  # also - we should make the md5checksums for each file compared to the
  # CDC website
  tarball_ending = grepl("[.]tar[.]bz2", id)
  files[!tarball_ending] = paste0(files[!tarball_ending], ".tar.bz2")
  urls = paste0("https://ftp.cdc.gov/pub/", version, "/", files)
  outfiles = sapply(urls, function(x) {
    destfile = file.path(exdir, basename(x))
    curl::curl_download(x, destfile = destfile, ..., quiet = quiet)
  })
  outfiles
  # httr::set_config(config(ssl_verifypeer = 0L))

}

write_data = function(data, file, format = "parquet") {
  path = tempfile()
  # get around bug for {i}
  args = list(
    data,
    path = path,
    format = format
  )
  if (format == "parquet") {
    args = c(args,
             list(# snappy or gzip allowed for bigquery
               compression = "gzip",
               compression_level = 9,
               allow_truncated_timestamps = FALSE))
  }
  do.call(arrow::write_dataset, args = args)
  x = list.files(path = path,
                 pattern = "part.*",
                 full.names = TRUE)
  stopifnot(length(x) == 1)
  ifile = sub(".gz$", "", file)
  fs::file_move(x, ifile)
  # file.rename(x, normalizePath(file))
  return(file)
}

vroom_write_csv = function(..., num_threads = 2) {
  vroom::vroom_write(..., delim = ",", num_threads = num_threads)
}

make_folds = function(data, nfolds) {
  n = nrow(data)
  data$fold = rep(1:nfolds, each = ceiling(n/nfolds))[1:n]
  stopifnot(!anyNA(data$fold))
  return(data)
}

check_time_diffs = function(time, sample_rate = 80) {
  dtime = as.numeric(diff(time, units = "secs"))
  rm(time)
  gc()
  mdtime = mean(dtime)
  dtime = unique(dtime)
  eps = 0.000001
  assertthat::assert_that(
    all(dtime > 0),
    all(dtime < 1),
    mdtime <= (1/sample_rate + eps)
  )
}

make_meta_df_from_files = function(files) {
  have_log = any(grepl("_log", files, ignore.case = TRUE))
  csv_files = files[!grepl("_log", files, ignore.case = TRUE)]
  csv_files = csv_files %>%
    stringr::str_replace("^[.]/", "")
  csvs = csv_files %>%
    stringr::str_replace("GT3XPLUS-AccelerationCalibrated-", "") %>%
    stringr::str_replace("-000-P000.*", "")
  csvs = strsplit(csvs, ".", fixed = TRUE)
  df = do.call(rbind, csvs)
  colnames(df) = c("firmware", "serial", "date")
  df = tibble::as_tibble(df) %>%
    dplyr::mutate(firmware = stringr::str_replace_all(firmware, "x", "."),
                  date = lubridate::ymd_hms(date),
                  file = csv_files)
  n_files = nrow(df)
  stopifnot(!anyNA(df$date))
  stopifnot(!anyNA(df))

  stopifnot(!anyNA(df$date))
  df$dtime = c(0, diff(df$date, units = "mins"))
  df$have_log = have_log
  df
}


get_meta_df = function(raw) {

  dir.create(dirname(raw), recursive = TRUE, showWarnings = FALSE)
  files = untar(raw, list = TRUE, verbose = FALSE,
                exdir = ".")
  df = make_meta_df_from_files(files)


  df
}

tarball_df = function(
    tarball_file,
    log_file,
    meta_file,
    num_threads = 1,
    ...) {
  ds = getOption("digits.secs")
  on.exit({
    options(digits.secs = ds)
  }, add = TRUE)
  options(digits.secs = 3)
  dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)
  dir.create(dirname(meta_file), showWarnings = FALSE, recursive = TRUE)
  tdir = tempfile()

  # create a temporary directory to put the unzipped data
  dir.create(tdir, showWarnings = TRUE)
  exit_code = untar(tarfile = tarball_file, exdir = tdir, verbose = TRUE)
  stopifnot(exit_code == 0)

  # List out the files
  files = list.files(path = tdir, full.names = FALSE, recursive = TRUE)
  # Create metadata dataset that puts all the hourly files into a df
  meta_df = make_meta_df_from_files(files)
  readr::write_csv(meta_df, file = meta_file, num_threads = num_threads)

  # get all the files in the tarball
  files = list.files(path = tdir, full.names = TRUE)
  # logs are different
  included_log_file = files[grepl("_log", files, ignore.case = TRUE)]
  stopifnot(length(included_log_file) <= 1)
  if (length(included_log_file) == 1) {
    R.utils::gzip(included_log_file, destname = log_file,
                  remove = FALSE, overwrite = TRUE,
                  compression = 9)
  }
  csv_files = files[!grepl("_log", files, ignore.case = TRUE)]
  # Read in the Data:   HEADER_TIMESTAMP, X, Y, Z (col_types_80hz)
  df = vroom::vroom(csv_files, num_threads = num_threads,
                    col_types = col_types_80hz)
  attr(df, "log_file") = log_file
  attr(df, "meta_df") = meta_df
  attr(df, "data_dir") = tdir

  return(df)
}

write_csv_gz = function(
    df, file,
    num_threads = 1L,
    ...) {
  dir.create(dirname(file),
             showWarnings = FALSE,
             recursive = TRUE)
  conn = gzfile(file, compression = 9, open = "wb")
  readr::write_csv(
    df,
    conn,
    ...,
    num_threads = num_threads)
  close(conn)
  return(file)
}

tarball_to_csv = function(tarball_file,
                          csv_file,
                          log_file,
                          meta_file,
                          num_threads = 2,
                          cleanup = TRUE,
                          ...) {
  ds = getOption("digits.secs")
  on.exit({
    options(digits.secs = ds)
  }, add = TRUE)
  options(digits.secs = 3)

  # print("Number of threads: ", num_threads)
  dir.create(dirname(csv_file), showWarnings = FALSE, recursive = TRUE)
  message("creating a df")
  df = tarball_df(tarball_file = tarball_file,
                  log_file = log_file,
                  meta_file = meta_file,
                  num_threads = num_threads,
                  ...)
  data_dir = attr(df, "data_dir")
  if (!is.null(data_dir) && cleanup) {
    on.exit(unlink(data_dir, recursive = TRUE), add = TRUE)
  }
  message(paste0("writing csv: ", csv_file))
  file = csv_file
  # if (have_pigz()) {
  #   cmd = paste0(
  #     "pigz -9 ",
  #     ifelse(
  #       !is.null(num_threads),
  #       paste0("--processes ", num_threads),
  #       ""),
  #     " > ", csv)
  #   file = pipe(cmd)
  # }

  print(head(df, 5))
  stopifnot(!all(as.numeric(df$HEADER_TIMESTAMP) %% 1 == 0))
  # need this because of return(x), no duplicate
  df = vroom::vroom_write(df, file = file, delim = ",",
                          num_threads = num_threads)
  message("CSV written")

  for (i in 1:3) gc()
  return(df)
}

summarize_meta_df = function(df, raw = NULL,
                             num_threads = 1L) {

  # adding zero so they are the same length for which later
  stopifnot(all(df$dtime <= 60))
  rtime = range(df$date)
  first_filename_time = rtime[1]
  last_filename_time = rtime[2]
  first_file = which(df$date == first_filename_time)
  last_file = which(df$date == last_filename_time)


  firmware = unique(df$firmware)
  stopifnot(length(firmware) == 1)
  serial = unique(df$serial)
  stopifnot(length(serial) == 1)

  have_log = unique(df$have_log)
  stopifnot(length(have_log) == 1)


  extract_files = df$file[c(first_file, last_file)]

  if (!is.null(raw) && file.exists(raw)) {
    tdir = tempfile()
    dir.create(tdir, showWarnings = TRUE)
    untar(tarfile = raw, files = paste0("./", extract_files), exdir = tdir)
    ds = getOption("digits.secs")
    on.exit({
      options(digits.secs = ds)
    }, add = TRUE)
    options(digits.secs = 3)
    dates = lapply(file.path(tdir, extract_files), readr::read_csv,
                   num_threads = num_threads)
    dates = lapply(dates, function(x) range(x$HEADER_TIMESTAMP))
    dates = unlist(dates)
    dates = as.POSIXct(dates, origin = lubridate::origin)
    rtime = as.character(range(dates))
    start_time = rtime[1]
    stop_time = rtime[2]
    unlink(tdir, recursive = TRUE)
  } else {
    start_time = lubridate::as_date(NA)
    stop_time = lubridate::as_date(NA)
  }
  df = tibble::tibble(
    start_time = start_time,
    stop_time = stop_time,
    first_filename_time = first_filename_time,
    last_filename_time = last_filename_time,
    firmware = firmware,
    serial = serial,
    have_log = have_log
  )
  df = as.data.frame(df)
  df
}

get_meta_summary_df = function(raw) {
  id = sub("[.].*", "", basename(raw))
  df = get_meta_df(raw)
  df = summarize_meta_df(df, raw)
  df$id = id
  df = df %>%
    dplyr::select(id, dplyr::everything())
  return(df)
}
meta_df = function(raw, curr_ids = NULL) {

  id = sub("[.].*", "", basename(raw))
  df = NULL
  if (is.null(curr_ids) || !id %in% curr_ids) {
    df = get_meta_summary_df(raw)
  }
  df
}


read_80hz = function(file, num_threads = 1, ...) {

  message("Reading in Full data")
  dat = vroom::vroom(file,
                     col_types = col_types_80hz,
                     num_threads = num_threads,
                     ...)
  probs = vroom::problems(dat)
  stopifnot(nrow(probs) == 0)
  readr::stop_for_problems(dat)
  dat
}
# pad_80hz = function(df) {
#
# }

col_time_with_frac_secs = function(...) {
  vroom::col_time(format = "%H:%M:%OS", ...)
}

read_log = function(file, ...) {
  vroom::vroom(file,
               col_types =
                 cols(
                   DAY_OF_DATA = col_double(),
                   START_TIME = col_time_with_frac_secs(),
                   END_TIME = col_time_with_frac_secs(),
                   DATA_QUALITY_FLAG_CODE = col_character(),
                   DATA_QUALITY_FLAG_VALUE = col_double()
                 ),
               ...)
}

header_to_day = function(df) {
  df %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      day = lubridate::floor_date(HEADER_TIMESTAMP),
      day = difftime(day, day[1], units = "days"),
      day = as.numeric(day),
      time = format(HEADER_TIMESTAMP, format = "%H:%M:%OS3")
    )
}



# process_nhanes_80hz = function(id, version,
#                                sample_rate = 80L,
#                                dynamic_range = c(-6L, 6L),
#                                verbose = TRUE
# ) {
#   stopifnot(length(id) == 1)
#   stopifnot(length(version) == 1)
#   file = file.path(version, "csv", paste0(id, ".csv.gz"))
#   counts_file = file.path(version, "counts", paste0(id, ".csv.gz"))
#   measures_file = file.path(version, "summary", paste0(id, ".csv.gz"))
#   meta_file = file.path(version, "meta", paste0(id, ".csv.gz"))
#   log_file = file.path(version, "logs", paste0(id, ".csv.gz"))
#   files = c(file, counts_file, measures_file, meta_file, log_file)
#   sapply(files, function(x) {
#     dir.create(dirname(x), showWarnings = FALSE, recursive = TRUE)
#   })
#
#   message("downloading tarball")
#   tarball = download_80hz(id, version, quiet = verbose < 2)
#   # this runs checks on firmware and diff time of 60 mins
#   message("Making tarball df")
#   df = tarball_df(tarball, cleanup = TRUE, num_threads = 1,
#                   outfile = file)
#   tmp_log_file = attr(df, "log_file")
#   file.copy(tmp_log_file, log_file, overwrite = TRUE)
#
#   id_meta_df = attr(df, "meta_df")
#   meta_df = summarize_meta_df(id_meta_df, raw = NULL)
#   meta_df$start_time = as.character(min(df$HEADER_TIMESTAMP))
#   meta_df$stop_time = as.character(max(df$HEADER_TIMESTAMP))
#   meta_df$id = id
#   meta_df = meta_df %>%
#     dplyr::select(id, dplyr::everything())
#
#   readr::write_csv(meta_df, meta_file)
#
#   # run quick checks
#   # Add to database!!!
#   stopifnot(
#     lubridate::is.POSIXct(df$HEADER_TIMESTAMP)
#   )
#
#   check_time_diffs(df$HEADER_TIMESTAMP, sample_rate = sample_rate)
#
#   run_time = system.time({
#     measures = SummarizedActigraphy::calculate_measures(
#       df,
#       calculate_ac = FALSE,
#       fix_zeros = FALSE,
#       fill_in = FALSE,
#       trim = FALSE,
#       dynamic_range = dynamic_range,
#       calculate_mims = FALSE,
#       flag_data = FALSE,
#       sample_rate = sample_rate,
#       verbose = verbose > 0)
#   })
#   measures = tibble::as_tibble(measures)
#   if (!"HEADER_TIMESTAMP" %in% colnames(measures)) {
#     measures = measures %>%
#       dplyr::rename(HEADER_TIMESTAMP = time)
#   }
#   measures$id = id
#   measures = measures %>%
#     dplyr::select(id, dplyr::everything())
#
#   readr::write_csv(measures, measures_file, num_threads = 1)
#
#   rm(df)
#   for (i in 1:10) gc()
#
#   counts = agcounter::convert_counts_csv(
#     file,
#     outfile = counts_file,
#     sample_rate = sample_rate,
#     epoch_in_seconds = 60L,
#     verbose = 2,
#     time_column = "HEADER_TIMESTAMP")
#
#   counts$id = id
#   counts = counts %>%
#     dplyr::select(id, dplyr::everything()) %>%
#     rename(AC_X = X, AC_Y = Y, AC_Z = Z)
#   counts = as.data.frame(counts)
#
#   readr::write_csv(counts, counts_file, num_threads = 1)
#
#   # unlink(dirname(tmp_log_file), recursive = TRUE)
#   file.remove(tarball)
#   list(
#     csv_file = file,
#     measures = measures,
#     counts = counts,
#     id_meta_df = id_meta_df,
#     meta_df = meta_df,
#     counts_file = counts_file,
#     measures_file = measures_file,
#     meta_file = meta_file,
#     log_file = log_file
#   )
# }




floor_date2 = function(x, ...) {
  if (hms::is_hms(x)) {
    x = lubridate::hms(x)
  }
  x %>%
    lubridate::as_datetime() %>%
    lubridate::floor_date(...) %>%
    hms::as_hms()
}

make_flag_df = function(id, version, ...) {
  log_file = file.path(version, "logs", paste0(id, ".csv.gz"))
  csv_file = file.path(version, "csv", paste0(id, ".csv.gz"))

  if (!file.exists(csv_file)) {
    gcs_download(csv_file)
    all_minutes = read_80hz(csv_file, ...)
    file.remove(csv_file)
  } else {
    all_minutes = read_80hz(csv_file, ...)
  }
  message("Getting all_minutes")
  all_minutes = all_minutes$HEADER_TIMESTAMP
  if (is.unsorted(all_minutes) || anyNA(all_minutes)) {
    all_minutes = range(all_minutes)
  } else {
    all_minutes = c(all_minutes[1], all_minutes[length(all_minutes)])
  }
  all_minutes = lubridate::floor_date(all_minutes, "1 minute")
  all_minutes = seq(all_minutes[1], all_minutes[2], by = 60L)
  all_minutes = unique(all_minutes)

  if (!file.exists(log_file)) {
    gcs_download(log_file, overwrite = TRUE)
  }
  log = read_log(log_file, ...) %>%
    janitor::clean_names()
  log = log %>%
    arrange(day_of_data, start_time)
  log = log %>%
    mutate(
      start_minute = floor_date2(start_time, unit = "minutes"),
      end_minute = floor_date2(end_time, unit = "minutes")
    )

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
  df = tibble::tibble(
    HEADER_TIME_STAMP = all_minutes
  )
  min_day = min(df$HEADER_TIME_STAMP)
  min_day = lubridate::floor_date(min_day, unit = "days")
  min_day = lubridate::as_date(min_day)

  flags = matrix(FALSE, nrow = nrow(df), ncol = length(flag_types))
  colnames(flags) = flag_types
  flags = tibble::as_tibble(flags)
  flags = dplyr::bind_cols(df, flags)

  if (nrow(log) > 0) {

    flags = flags %>%
      tidyr::gather(flag, value = value, -HEADER_TIME_STAMP) %>%
      select(-value)


    log = log %>%
      mutate(
        date = min_day + day_of_data - 1,
        start_dt = paste0(as.character(date), " ", as.character(start_time)),
        end_dt = paste0(as.character(date), " ", as.character(end_time))
      )
    log = log %>%
      dplyr::mutate(
        start_dt = lubridate::ymd_hms(start_dt),
        end_dt = lubridate::ymd_hms(end_dt))

    # assuming all the diff times are supposed to be next day
    # add_day = lubridate::hour(log$start_time) >= 20 &
    #   lubridate::hour(log$end_time) < 20
    add_day = log$end_dt < log$start_dt
    if (any(add_day)) {
      message(paste0("Weird dates for id ", idf$id))
    }
    log$end_dt[add_day] = log$end_dt[add_day] + lubridate::as.period(1, "day")

    log = log %>%
      dplyr::mutate(
        start_dt = lubridate::floor_date(start_dt, "minutes"),
        end_dt = lubridate::floor_date(end_dt, "minutes"))
    log = log %>%
      select(data_quality_flag_code, start_dt, end_dt) %>%
      distinct()

    log_df = mapply(function(start, end, flag) {
      times = seq(start, end, by = 60L)
      ddf = tibble::tibble(
        HEADER_TIME_STAMP = times,
        flag = flag,
        value = TRUE
      )
    }, log$start_dt, log$end_dt,
    log$data_quality_flag_code, SIMPLIFY = FALSE)

    log_df = dplyr::bind_rows(log_df)

    flags = left_join(flags, log_df)
    flags = flags %>%
      tidyr::spread(flag, value = value, fill = FALSE)

  }
  return(flags)

}



















filetype <- function(path){
  f = file(path)
  ext = summary(f)$class
  close.connection(f)
  ext
}
is_gzip = function(path) {
  filetype(path) %in% c("gzfile", "xzfile", "bzfile")
}


rename_xyzt = function(csv_file) {
  tfile = tempfile(fileext = ".csv")
  if (is_gzip(csv_file)) {
    R.utils::gunzip(csv_file, destname = tfile, remove = FALSE)
  } else {
    file.copy(csv_file, tfile)
  }
  qtfile = shQuote(tfile)
  # cmd = paste0("zcat ", shQuote(csv_file), " > ", qtfile)
  # system(cmd)
  cmd = paste0("sed -ibak '1s/HEADER_TIMESTAMP/time/' ", qtfile)
  system(cmd)
  cmd = paste0("sed -ibak '1s/X/x/' ", qtfile)
  system(cmd)
  cmd = paste0("sed -ibak '1s/Y/y/' ", qtfile)
  system(cmd)
  cmd = paste0("sed -ibak '1s/Z/z/' ", qtfile)
  system(cmd)
  cmd = paste0("sed -ibak 's/T/ /' ", qtfile)
  system(cmd)
  cmd = paste0("sed -ibak 's/Z//' ", qtfile)
  system(cmd)
  return(tfile)
}

summarise_nhanes_80hz = function(
    csv_file,
    log_file,
    meta_file,
    counts_file,
    measures_file,
    summary_meta_file = NULL,
    sample_rate = 80L,
    dynamic_range = c(-6L, 6L),
    num_threads = 1L,
    verbose = TRUE
) {
  X = Y = Z = NULL
  rm(list = c("X", "Y", "Z"))
  id = basename(csv_file)
  id = sub(".csv.*", "", id)
  files = c(csv_file, counts_file, measures_file, meta_file, log_file)
  sapply(files, function(x) {
    dir.create(dirname(x), showWarnings = FALSE, recursive = TRUE)
  })

  ds = getOption("digits.secs")
  on.exit({
    options(digits.secs = ds)
  }, add = TRUE)
  options(digits.secs = 3)

  df = read_80hz(csv_file, num_threads = num_threads)
  print(head(df))

  id_meta_df = readr::read_csv(meta_file, num_threads = num_threads)
  meta_df = summarize_meta_df(id_meta_df, raw = NULL)
  rtime = range(df$HEADER_TIMESTAMP, na.rm = TRUE)
  meta_df$start_time = as.character(rtime[1])
  meta_df$stop_time = as.character(rtime[2])
  meta_df$id = id
  meta_df = meta_df %>%
    dplyr::select(id, dplyr::everything())

  # run quick checks
  # Add to database!!!
  stopifnot(
    lubridate::is.POSIXct(df$HEADER_TIMESTAMP)
  )

  check_time_diffs(df$HEADER_TIMESTAMP, sample_rate = sample_rate)

  run_time = system.time({
    measures = SummarizedActigraphy::calculate_measures(
      df,
      calculate_ac = FALSE,
      fix_zeros = FALSE,
      fill_in = FALSE,
      trim = FALSE,
      dynamic_range = dynamic_range,
      calculate_mims = FALSE,
      flag_data = FALSE,
      sample_rate = sample_rate,
      verbose = verbose > 0)
  })
  measures = tibble::as_tibble(measures)
  if (!"HEADER_TIMESTAMP" %in% colnames(measures)) {
    measures = measures %>%
      dplyr::rename(HEADER_TIMESTAMP = time)
  }
  measures$id = id
  measures = measures %>%
    dplyr::select(id, dplyr::everything())

  readr::write_csv(measures, measures_file, num_threads = 1)

  rm(df)
  for (i in 1:10) gc()

  # From muschellij2/agcounter
  counts = agcounter::convert_counts_csv(
    csv_file,
    outfile = counts_file,
    sample_rate = sample_rate,
    epoch_in_seconds = 60L,
    verbose = 2,
    time_column = "HEADER_TIMESTAMP")

  counts$id = id
  counts = counts %>%
    dplyr::select(id, dplyr::everything()) %>%
    dplyr::rename(AC_X = X, AC_Y = Y, AC_Z = Z)
  counts = as.data.frame(counts)

  readr::write_csv(counts, counts_file, num_threads = 1)

  list(
    csv_file = csv_file,
    measures = measures,
    counts = counts,
    id_meta_df = id_meta_df,
    meta_df = meta_df,
    counts_file = counts_file,
    measures_file = measures_file,
    meta_file = meta_file,
    log_file = log_file
  )
}

