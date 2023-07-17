library(dplyr)
library(tidyr)
library(agcounts)
source("helper_functions.R")
options(digits.secs = 3)
bucket = "nhanes_80hz"
bucket_setup(bucket)
config = trailrun::cr_gce_setup(bucket = bucket)

wide = readRDS("wide.rds")
wide = get_wide_data()
readr::write_rds(wide, "wide.rds")

nfolds = 100
wide = make_folds(wide, nfolds)
if ("counts" %in% colnames(wide)) {
  wide$run = !is.na(wide$counts)
} else {
  wide$run = FALSE
}

wide %>%
  count(run,fold) %>%
  spread(run, n)
cat(wide$id[!wide$run & wide$fold %in% c(95)], sep = " ")
