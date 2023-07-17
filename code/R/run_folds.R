library(googleCloudRunner)
library(googleCloudStorageR)
library(trailrun)
library(dplyr)
source("helper_functions.R")
source("run_helpers.R")
options("googleCloudStorageR.format_unit" = "b")
bucket = "nhanes_80hz"
bucket_setup(bucket)
version = "pax_h"
trailrun::cr_gce_setup(bucket = bucket)

name = trailrun::streamline_private_image("nhanes_80hz")

# file = "make_meta.R"
file = "process_single_id.R"


pre_steps = c(
  googleCloudRunner::cr_buildstep("docker",
                                  c("pull",
                                    name),
                                  id = "image_pull")
)

my_seq = c(71)
n_tasks = 1
groups = 0:(length(my_seq)-1) %/% n_tasks + 1
tasklist = split(my_seq, groups)
tasks = tasklist[[1]]
lapply(tasklist, function(tasks) {
  all_steps = lapply(tasks, function(r) {
    steps = googleCloudRunner::cr_buildstep(
      name = name,
      entrypoint = "R",
      args = c("-f", file),
      env = c(paste0("TASK_ID=", r),
              paste0("VERSION=", version)),
      waitFor = "image_pull"
    )
  })
  all_steps = unname(all_steps)
  steps = sapply(all_steps, c)

  steps = c(pre_steps, steps)
  names(steps) = NULL
  steps

  cores = 32
  gb_amount = "200"
  options = list(diskSizeGb = gb_amount)
  if (cores > 4) options$machineType = paste0("N1_HIGHCPU_", cores)
  # googleAuthR::gar_gce_auth
  trailrun::cr_run_github(steps, path = ".", timeout = 3600L * 6L,
                          options = options, launch_browser = FALSE)
})
