library(googleCloudRunner)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
source("helper_functions.R")
source("run_helpers.R")
options("googleCloudStorageR.format_unit" = "b")
version = "pax_g"
bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)
id_file = paste0(version, "_ids.txt")
ids = readLines(id_file)

# if (!file.exists("wide.rds")) {
#   wide = get_wide_data()
#   readr::write_rds(wide, "wide.rds")
# } else {
#   wide = readr::read_rds("wide.rds")
# }
# wide[is.na(wide$csv),]

gcs = googleCloudStorageR::gcs_list_objects()
gcs = gcs %>%
  rename(file = name)
bad = gcs %>%
  filter(size_bytes == 0 & !grepl("/$", file))
stopifnot(nrow(bad) == 0)
df = readr::read_rds(paste0(version, "_filenames.rds"))

ddf = left_join(df, gcs, by = "file") %>%
  select(id, file, size_bytes, bytes) %>%
  mutate(same = bytes == size_bytes)
ddf %>% filter(!same | is.na(same))
ddf %>% filter(size_bytes==0)

pigz_vroom = function(file, ...) {
  vroom::vroom(pipe(paste0("pigz -dc ", file)))
}
already_run = df$logfile %in% gcs$file & df$full_csv %in% gcs$file
ids_already_run = df[already_run, ]$id
# ids = setdiff(ids, ids_already_run)

# steps = run_ids(
#   version = version,
#   ids = sample(ids, 90),
#   file_to_run = "make_csvs.sh")
# gb_amount = "200"
# yaml = cr_build_yaml(steps, timeout = 3600*3,
#                      options = list(diskSizeGb = gb_amount))
# cr_build(yaml, launch_browser = FALSE)

# ids = ids[1:(min(2000, length(ids)))]

##########################
# Big Run
##########################
# steps = make_parallel_steps(
#   version = version,
#   ids = ids[2000:3000],
#   nfolds = 4,
#   file_to_run = "make_csvs.sh")
#
# # steps = pre_steps
# gb_amount = "200"
# yaml = cr_build_yaml(
#   steps,
#   timeout = 3600*3,
#   options = list(
#     machineType = "N1_HIGHCPU_8",
#     diskSizeGb = gb_amount
#     ))
# cr_build(yaml, launch_browser = FALSE)


# n_per_6_hours = 300 # 8 core
cores = 32
n_per_fold = ifelse(cores > 4, 50, 10) # fixed for 6 hours
n = length(ids)
folds_per_run = max(2, cores / 4)
n_per_run = n_per_fold * folds_per_run
nruns = ceiling(n/n_per_run)
nfolds = ceiling(folds_per_run * nruns)
folds = seq(nfolds)
runs = rep(1:nruns, each = folds_per_run)
split_folds = split(folds, runs)
# folds_to_run = split_folds[[1]]
lapply(split_folds, function(folds_to_run) {
  steps = make_parallel_steps(
    version = version,
    ids = ids,
    nfolds = nfolds,
    folds_to_run = folds_to_run,
    file_to_run = "make_csvs.sh")

  # steps = pre_steps
  gb_amount = "200"
  # yaml = cr_build_yaml(steps, timeout = 3600*3,
  #                      options = list(diskSizeGb = gb_amount))
  options = list(diskSizeGb = gb_amount)
  if (cores > 4) options$machineType = paste0("N1_HIGHCPU_", cores)

  yaml = cr_build_yaml(
    steps,
    timeout = 3600*6,
    options = options)
  cr_build(yaml, launch_browser = FALSE)
})
# 62171
