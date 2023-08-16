library(googleCloudRunner)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
library(tidyr)
library(vroom)
source("helper_functions.R")
source("run_helpers.R")
options("googleCloudStorageR.format_unit" = "b")
version = "pax_h"
options(digits.secs = 3)
bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)

if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}

ifile = 2
wide = wide[!is.na(wide$logs), ]
all_types = pbapply::pblapply(wide$logs, function(log_file) {
  if (!file.exists(log_file)) {
    gcs_download(log_file)
  }
  log = read_log(log_file, progress = FALSE) %>%
    janitor::clean_names()
  unique(log$data_quality_flag_code)
})
types = sort(unique(unlist(c(all_types))))
