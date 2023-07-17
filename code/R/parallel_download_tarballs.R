library(googleCloudRunner)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
source("helper_functions.R")
source("run_helpers.R")
options("googleCloudStorageR.format_unit" = "b")
version = "pax_h"
bucket = "nhanes_80hz"
bucket_setup(bucket)
trailrun::cr_gce_setup(bucket = bucket)
id_file = paste0(version, "_ids.txt")
ids = readLines(id_file)

gcs = googleCloudStorageR::gcs_list_objects()
gcs = gcs %>%
  rename(file = name)
df = readr::read_rds(paste0(version, "_filenames.rds"))

ddf = left_join(df, gcs, by = "file") %>%
  select(id, file, size_bytes, bytes) %>%
  mutate(same = bytes == size_bytes)
ddf %>% filter(!same | is.na(same))

ids_already_run = basename(gcs$file[grepl("\\.tar", gcs$file)])
ids_already_run = sub(".tar.bz2", "", ids_already_run)

# ids = setdiff(ids, ids_already_run)
files = file.path("raw", version, paste0(ids, ".tar.bz2"))
# ids = ids[!files %in% gcs$name]



steps = make_parallel_steps(version = version,
                            ids = ids,
                            nfolds = 32,
                            file_to_run = "get_tarballs.sh")

# steps = pre_steps
yaml = cr_build_yaml(steps, timeout = 1000)
yaml = cr_build_yaml(
  steps,
  timeout = 3600*6,
  options = list(machineType = "N1_HIGHCPU_32"))
cr_build(yaml)
