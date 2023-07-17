library(bigrquery)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
library(streamliner)
options(digits.secs = 3)
source("helper_functions.R")
bucket = "nhanes_80hz"
bucket_setup(bucket)
config = trailrun::cr_gce_setup(bucket = bucket)


if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}

# table = paste0(version, "_csvcheck")
# out = get_bqt_ids(table, dataset = bucket, project = config$project)
# bqt = out$bqt
# curr_ids = out$curr_ids


ids_file = "headers_checked.txt"
if (!file.exists(ids_file)) {
  curr_ids = NULL
} else {
  curr_ids = readLines(ids_file)
  curr_ids = unique(curr_ids)
}

wide = wide[ !wide$csv %in% curr_ids, ]
wide %>% count(version, is.na(csv))

version = Sys.getenv("VERSION", unset = NA)
if (!is.na(version)) {
  print(paste0("version is: ", version))
  wide = wide[ wide$version %in% version, ]
}

wide = make_folds(wide, nfolds = 100)

fold = as.numeric(Sys.getenv("TASK_ID", unset = NA))
if (!all(is.na(fold))) {
  print(paste0("Fold is: ", fold))
  wide = wide[ wide$fold %in% fold, ]
}


iid = 1

pigz_exists = nzchar(Sys.which("pigz"))
for (iid in seq(nrow(wide))) {
  idf = wide[iid,]
  csv_file = idf$csv
  id = idf$id
  print(id)
  cmd = paste0("./fix_headers.sh ", csv_file)
  if (!is.na(csv_file)) {
    system(cmd)
  }
}
