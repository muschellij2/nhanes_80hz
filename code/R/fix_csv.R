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

bad_ids = readLines("bad_ids.txt")

version = Sys.getenv("VERSION", unset = NA)
if (is.na(version)) {
  version = "pax_g"
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

fix_csv = function(id, measure_1min_bqt) {
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


    if (!file.exists(gz_file)) {
      gcs_download(gz_file)
    }
    xlines = readLines(gz_file, n = (80*60*60) + 1)
    index = grep("HEADER", xlines)
    bad_file = length(index) > 1 ||
      length(index) == 0
    rm(xlines)
    print_memory()
    if (bad_file) {
      message(id, " has messed up CSV")
      id_tarball_to_csv_fix(id, id_version, gz_file)
    }
    file.remove(gz_file)
  }
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
    fix_csv(id, measure_1min_bqt)
  } else {
    message("found bad id: ", id)
  }
  if ("lsos" %in% ls()) print(lsos())
}

