library(bigrquery)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
library(tidyr)
library(stringr)
library(streamliner)
options(digits.secs = 3)
source("helper_functions.R")
source("run_helpers.R")
options("googleCloudStorageR.format_unit" = "b")
version = Sys.getenv("VERSION", unset = NA)
if (is.na(version)) {
  version = "pax_h"
  print(paste0("version is: ", version))
}

bucket = "nhanes_80hz"
project = "streamline-resources"
if ("project" %in% formalArgs(trailrun::cr_gce_setup)) {
  config = trailrun::cr_gce_setup(project = project, bucket = bucket)
} else {
  config = trailrun::cr_gce_setup(bucket = bucket)
}
bucket_setup(bucket, project = project)

if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}

bad_ids = readLines("bad_ids.txt")


data = wide[ wide$version %in% version,]
table = paste0(version, "_meta")
out = get_bqt_ids(table, dataset = bucket, project)
curr_ids = out$curr_ids
bqt = out$bqt

nfolds = 100
data = make_folds(data, nfolds)
data$run = data$id %in% curr_ids
print(version)
if (any(!data$run)) {
  data %>%
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
  data = data[ data$fold %in% fold, ]
}

create_db_entry = function(raw, curr_ids = NULL) {
  df = meta_df(raw, curr_ids)
  if (!is.null(df)) {
    push_table_up(bqt, df)
  }
  return(invisible(df))
}

# data = data[!data$run,]
raw = data$raw[1]
# raw has an issue "pax_g/raw/65108.tar.bz2"
for (raw in data$raw) {
  print(raw)
  id = sub("[.].*", "", basename(raw))
  if (!id %in% bad_ids) {
    create_db_entry(raw, curr_ids)
  } else {
    message("found bad id: ", id)
  }
}


