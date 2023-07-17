library(googleCloudStorageR)
library(arrow)
library(magrittr)
source("helper_functions.R")
version = "pax_h"

bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)

index = 1

process_index = function(index, cleanup = TRUE,
                         upload = TRUE) {

  stopifnot(length(index) == 1)
  xdf = df = readr::read_rds(paste0(version, "_filenames.rds"))

  gcs = googleCloudStorageR::gcs_list_objects()
  gcs = gcs %>%
    dplyr::rename(file = name)

  idf = df[index,]
  idf$index = index
  print(idf)
  file = idf$file
  parquet = idf$parquet
  logfile = idf$logfile
  full_csv = idf$full_csv
  all_files = c(file, parquet, logfile, full_csv)
  if (all(all_files %in% gcs$file) && cleanup) {
    file.remove(all_files)
  }
  if (!all(all_files %in% gcs$file)) {
    if (!file %in% gcs$file) {
      if (!file.exists(file)) {
        dir.create(dirname(idf$file), showWarnings = FALSE,
                   recursive = TRUE)
        curl::curl_download(idf$url, idf$file)
      }
      if (upload) {
        googleCloudStorageR::gcs_upload(file = file,
                                        name = file)
      }
    }
    if (!parquet %in% gcs$file ||
        !logfile %in% gcs$file ||
        !full_csv %in% gcs$file) {
      if (!file.exists(file)) {
        dir.create(dirname(idf$file), showWarnings = FALSE,
                   recursive = TRUE)
        curl::curl_download(idf$url, idf$file)
      }
      exdir = tempfile()
      dir.create(exdir, showWarnings = FALSE)
      csv_files = untar(file, exdir = exdir)
      all_csvs = list.files(path = exdir,
                            pattern = ".csv",
                            ignore.case = TRUE,
                            full.names = TRUE)
      log_csv = all_csvs[grepl("logs", ignore.case = TRUE, all_csvs)]
      all_csvs = all_csvs[grepl("sensor", all_csvs)]
    }
    dir.create(dirname(logfile), showWarnings = FALSE, recursive = TRUE)
    if (!logfile %in% gcs$file) {
      if (!file.exists(logfile)) {
        ff = pigz(log_csv)
        fs::file_move(ff, logfile)
      }
      stopifnot(file.exists(logfile))
      if (upload) {
        googleCloudStorageR::gcs_upload(file = logfile,
                                        name = logfile)
      }
    }

    if (!parquet %in% gcs$file ||
        !full_csv %in% gcs$file) {
      data = open_dataset(all_csvs, format = "csv")
      outdir = dirname(parquet)
      if (!dir.exists(outdir)) {
        dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
      }
      if (!parquet %in% gcs$file) {
        write_data(data, parquet)
        if (upload) {
          googleCloudStorageR::gcs_upload(file = parquet,
                                          name = parquet)
        }
      }
      outdir = dirname(full_csv)
      if (!dir.exists(outdir)) {
        dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
      }
      if (!full_csv %in% gcs$file) {
        write_data(data, full_csv, format = "csv")
        if (upload) {
          googleCloudStorageR::gcs_upload(file = full_csv,
                                        name = full_csv)
        }
      }
    }
  }
}

sapply(1:5, function(index) process_index(index))
