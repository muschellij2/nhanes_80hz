library(googleCloudRunner)
library(bigrquery)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
library(tidyr)
library(MIMSunit)
library(SummarizedActigraphy)
# needed for bqt tbl capability
library(streamliner)
options(digits.secs = 3)
source("helper_functions.R")

bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)
config = trailrun::cr_gce_setup(bucket = bucket)
run_start_time = Sys.time()

bad_ids = readLines("bad_ids.txt")

version = Sys.getenv("VERSION", unset = NA)
if (is.na(version)) {
  version = "pax_h"
}
#############################################
# Read in the Metadata
#############################################
data_bqt = bq_table(project = config$project,
                    dataset = bucket,
                    table = paste0(version, "_data"))

measure_1min_bqt = bq_table(project = config$project,
                            dataset = bucket,
                            table = paste0(version, "_measures_1min"))

bqt = bq_table(project = config$project,
               dataset = bucket,
               table = paste0(version, "_meta"))
meta_df = tbl(bqt) %>%
  collect()


##########################################
# Read in the wide data
##########################################
if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}

print(paste0("version is: ", version))
wide = wide[ wide$version %in% version, ]

wide = meta_df %>% select(id) %>% mutate(id = as.character(id)) %>%
  left_join(wide)
wide = wide %>%
  dplyr::arrange(id)

# data = wide[ wide$version %in% version, ]
nfolds = 40
wide = make_folds(wide, nfolds)
# wide$run = in_bqt(data_bqt, wide$id) && in_bqt(measure_1min_bqt, wide$id)
wide$run = in_bqt(measure_1min_bqt, wide$id)
print(version)
if (any(!wide$run)) {
  wide %>%
    count(run,fold) %>%
    spread(run, n) %>%
    filter(!is.na(`FALSE`)) %>%
    mutate(version = version)
} else {
  print("All have been run")
}

fold = as.numeric(Sys.getenv("TASK_ID"))
if (!is.na(fold)) {
  print(paste0("Fold is: ", fold))
  wide = wide[ wide$fold %in% fold, ]
}


print_memory = function() {
  for (i in 1:3) gc()
  if ("lsos" %in% ls()) print(lsos())
  print(pryr::mem_used())
}

process_csv = function(id, measure_1min_bqt) {
  run_data = wide[wide$id == id,] %>%
    distinct()
  id_version = run_data$version

  # has_been_run = in_bqt(data_bqt, id) && in_bqt(measure_1min_bqt, id)
  has_been_run = in_bqt(measure_1min_bqt, id)

  if (!has_been_run) {

    meta_idf = meta_df %>%
      filter(id %in% run_data$id) %>%
      distinct()
    start_time = meta_idf$start_time
    stop_time = meta_idf$stop_time
    rm(meta_idf)
    gz_file = run_data$csv
    outfile = file.path(version, "summary", basename(gz_file))

    if (!file.exists(outfile)) {
      if (!file.exists(gz_file)) {
        gcs_download(gz_file)
      }
      # should use pigz here
      csv_file = R.utils::gunzip(
        gz_file, temporary = TRUE, remove = FALSE,
        overwrite = TRUE)
      file.remove(gz_file)
      xlines = readLines(csv_file, n = (80*60*60) + 1)
      bad_file = length(grep("HEADER", xlines)) > 1
      rm(xlines)
      print_memory()
      if (bad_file) {
        message(id, " has messed up CSV")
        file.remove(csv_file)
        return(NULL)
        dat = id_tarball_to_csv_fix(id, id_version, gz_file)
      } else {
        dat = read_80hz(csv_file)
      }
      print_memory()
      file.remove(csv_file)
      file.remove(file.path(tempdir(), basename(csv_file)))
      # message("Running AGCounts")
      # for (i in 1:3) gc()
      # res = agcounter::get_counts_csv(
      #   csv_file,
      #   sample_rate = 80L,
      #   epoch_in_seconds = 60L,
      #   verbose = 2)
      # res = res %>%
      #   mutate(AC = sqrt((X^2 +Y^2 + Z^2)/3)) %>%
      #   select(HEADER_TIME_STAMP, AC)


      frac_seconds = as.numeric(dat$HEADER_TIMESTAMP) %% 1
      stopifnot(any(frac_seconds > 0))
      rm(frac_seconds)
      dat = dat %>%
        rename(HEADER_TIME_STAMP = HEADER_TIMESTAMP)
      print("renaming header")
      print_memory()


      # Add to database!!!
      stopifnot(
        lubridate::is.POSIXct(dat$HEADER_TIME_STAMP)
      )

      check_csv = function(time, start_time, stop_time) {
        # dat = vroom::vroom(csv_file)
        r = range(time)
        check_start = isTRUE(all.equal(r[1], start_time, tolerance = 0.99))
        if (!check_start) {
          print("r1")
          print(r[1])
          print(start_time)
        }
        check_stop = isTRUE(all.equal(r[2], stop_time, tolerance = 0.99))
        if (!check_stop) {
          print("r2")
          print(r[2])
          print(stop_time)
        }
        assertthat::assert_that(
          # weird conversion with bigrquery db
          check_start,
          check_stop,
          msg = "times are not right"
        )
        dtime = as.numeric(diff(time, units = "secs"))
        rm(time)
        gc()
        mdtime = mean(dtime)
        dtime = unique(dtime)
        eps = 0.000001
        assertthat::assert_that(
          all(dtime > 0),
          mdtime <= (1/80 + eps)
        )
        rm(dtime)
        gc()
        return(TRUE)
      }
      check_csv(time = dat$HEADER_TIME_STAMP, start_time, stop_time)
      print_memory()

      # message("Uploading Full data")
      # dat$id = id
      # dat = dat %>%
      #   select(id, everything())
      # push_table_up(data_bqt, dat)
      # dat$id = NULL

      dynamic_range = c(-6, 6)
      print("dynamic range")
      print_memory()
      # dat = data.table::as.data.table(dat)
      print_memory()

      run_time = system.time({
        measures = SummarizedActigraphy::calculate_measures(
          dat,
          calculate_ac = FALSE,
          fix_zeros = FALSE,
          fill_in = FALSE,
          trim = FALSE,
          dynamic_range = dynamic_range,
          calculate_mims = FALSE,
          flag_data = FALSE,
          sample_rate = 80L)
      })
      message(paste0("run time ", run_time["elapsed"]))
      rm(dat)
      print_memory()
      measures = tibble::as_tibble(measures)
      if (!"HEADER_TIME_STAMP" %in% colnames(measures)) {
        measures = measures %>%
          dplyr::rename(HEADER_TIME_STAMP = time)
      }
      print_memory()

      # message("Joining Measures")
      #
      # measures = full_join(res, measures)
      # rm(res)

      message("Uploading Measures")
      measures$id = id
      measures = measures %>%
        select(id, everything())

      message("Uploading Measures CSV")
      dir.create(dirname(outfile), showWarnings = FALSE, recursive = TRUE)
      readr::write_csv(measures, file = outfile)
    } else {
      measures = readr::read_csv(outfile)
    }

    time_change = as.numeric(difftime(Sys.time(),
                                      run_start_time, unit = "hours"))
    if (time_change >= 0.85) {
      message("refreshing token")
      googleAuthR::gar_deauth()
      trailrun::cr_gce_setup(bucket = bucket)
      x = googleAuthR::gar_token()
      x$auth_token$refresh()
    }
    push_table_up(measure_1min_bqt, measures)

    googleCloudStorageR::gcs_upload(outfile, bucket = bucket,
                                    name = outfile)
    print_memory()
    # file.remove(outfile)
  }
  # has_been_run = in_bqt(data_bqt, id) && in_bqt(measure_1min_bqt, id)
  has_been_run = in_bqt(measure_1min_bqt, id)
  stopifnot(has_been_run)
}


# validation the CSV shifting time (first or last) is the right call
# cor(measures$MIMS_UNIT, c(NA, measures$AC[seq(1,nrow(measures)-1)]),
#     use = "complete")
# cor(measures$MIMS_UNIT, measures$AC, use = "complete")
# cor(measures$AI, measures$AC, use = "complete")

# Add to database!!!
wide = wide[!wide$run & !is.na(wide$csv),]
id = wide$id[1]
for (id in wide$id) {
  print(id)
  if (!id %in% bad_ids) {
    process_csv(id, measure_1min_bqt)
  } else {
    message("found bad id: ", id)
  }
  if ("lsos" %in% ls()) print(lsos())
}

