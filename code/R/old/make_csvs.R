library(googleCloudRunner)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
source("helper_functions.R")
bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)

if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}
# 62164

# iid = which(wide$id == "62164")
iid = 1
for (iid in seq(nrow(wide))) {
  idf = wide[iid,]
  id = idf$id
  version = idf$version
  tarball = idf$raw
  csv = idf$csv
  id_tarball_to_csv(id, version, csv)
  gcs_upload_file(csv)
  for (i in 1:3) gc()
}
