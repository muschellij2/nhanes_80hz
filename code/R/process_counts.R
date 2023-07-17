library(googleCloudRunner)
library(bigrquery)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
library(tidyr)
library(agcounts)
# needed for bqt tbl capability
library(streamliner)
options(digits.secs = 3)
source("helper_functions.R")
version = "pax_g"
bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)
config = trailrun::cr_gce_setup(bucket = bucket)
run_start_time = Sys.time()

bad_ids = readLines("bad_ids.txt")

#############################################
# Read in the Metadata
#############################################
data_bqt = bq_table(project = config$project,
                    dataset = bucket,
                    table = paste0(version, "_data"))

count_1min_bqt = bq_table(project = config$project,
                          dataset = bucket,
                          table = paste0(version, "_counts_1min"))

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


wide = meta_df %>% select(id) %>% mutate(id = as.character(id)) %>%
  left_join(wide)
wide = wide %>%
  dplyr::arrange(id)

nfolds = 100
wide = make_folds(wide, nfolds)
wide$run = in_bqt(count_1min_bqt, wide$id)
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

# counter = reticulate::import_from_path("extract", path = ".")
# counter = counter$convert_counts_csv

process_counts = function(id, count_1min_bqt) {
  run_data = wide[wide$id == id,]

  # has_been_run = in_bqt(data_bqt, id) && in_bqt(measure_1min_bqt, id)
  has_been_run = in_bqt(count_1min_bqt, id)
  gz_file = run_data$csv
  outfile = file.path(version, "counts", basename(gz_file))
  dir.create(dirname(outfile), showWarnings = FALSE, recursive = TRUE)

  if (!has_been_run || (has_been_run && is.na(run_data$counts))) {

    meta_idf = meta_df %>%
      filter(id %in% run_data$id)
    start_time = meta_idf$start_time
    stop_time = meta_idf$stop_time
    rm(meta_idf)


    if (!file.exists(outfile)) {
      if (!file.exists(gz_file)) {
        gcs_download(gz_file)
      }
      xlines = readLines(gz_file, n = (80*60*60) + 1)
      bad_file = length(grep("HEADER", xlines)) > 1
      rm(xlines)
      if (bad_file) {
        message(id, " has messed up CSV")
        file.remove(gz_file)
        return(NULL)
      }
      message("Running AGCounts")
      res = agcounts::convert_counts_csv(
        gz_file,
        outfile = outfile,
        sample_rate = 80L,
        epoch_in_seconds = 60L,
        verbose = 2,
        time_column = "HEADER_TIMESTAMP")
      # res = counter(file = gz_file,
      #               outfile = outfile,
      #               freq = 80L, epoch = 60L,
      #               verbose = TRUE,
      #               time_column = "HEADER_TIMESTAMP")
      # system(paste0("python3 convert.py -i ", gz_file, " -o ", outfile))
      # %>%
      # select(HEADER_TIME_STAMP, AC)
    } else {
      res = readr::read_csv(outfile)
    }

    res$id = id
    res = res %>%
      dplyr::select(id, everything()) %>%
      dplyr::rename(HEADER_TIME_STAMP = HEADER_TIMESTAMP)
    res = as.data.frame(res)
    time_change = as.numeric(difftime(Sys.time(),
                                      run_start_time, unit = "hours"))
    if (time_change >= 0.85) {
      message("refreshing token")
      googleAuthR::gar_deauth()
      trailrun::cr_gce_setup(bucket = bucket)
      x = googleAuthR::gar_token()
      x$auth_token$refresh()
    }

    googleCloudStorageR::gcs_upload(outfile, bucket = bucket,
                                    name = outfile)

    if (!has_been_run) {
      message("Uploading Counts")
      push_table_up(count_1min_bqt, res)
    }
    print_memory()
    # file.remove(outfile)
  }
  # has_been_run = in_bqt(data_bqt, id) && in_bqt(measure_1min_bqt, id)
  has_been_run = in_bqt(count_1min_bqt, id)
  stopifnot(has_been_run)
}

# Add to database!!!
wide = wide[!wide$run | (wide$run & is.na(wide$counts)),]
id = wide$id[1]
for (id in wide$id) {
  print(id)
  if (!id %in% bad_ids) {
    process_counts(id, count_1min_bqt)
  } else {
    message("found bad id: ", id)
  }
}
