library(dplyr)
library(tidyr)
library(agcounter)
source("helper_functions.R")
options(digits.secs = 3)
bucket = "nhanes_80hz"
bucket_setup(bucket)
Sys.unsetenv("RETICULATE_PYTHON")
config = trailrun::cr_gce_setup(bucket = bucket)

# readLines("signal_nil_check.py")
# has_conda = !inherits(try(reticulate::conda_binary()), "try-error")
# if (!has_conda) {
#   x = try({reticulate::install_miniconda()})
#   if (inherits(x, "try-error")) {
#     Sys.sleep(60)
#   }
#   reticulate::conda_install(packages = c("pandas", "numpy", "scipy"))
# }
#
#
# # reticulate::miniconda_path()
# choose_right_conda = function(condaenv, conda = "auto") {
#   envs = reticulate:::conda_list(conda = conda)
#   matches = which(envs$name == condaenv)
#   envs = envs[matches,]
#   if (nrow(envs) > 1) {
#     if (any(grepl("builder", envs$python)) &
#         !all(grepl("builder", envs$python))) {
#       envs = envs[!grepl("builder", envs$python), ]
#     }
#     envs = envs[1,]
#   }
#   message(paste0("using python ", envs$python))
#   reticulate::use_python(python = envs$python)
# }
# x = try({choose_right_conda(condaenv = "r-reticulate")})
# ld = Sys.getenv("LD_LIBRARY_PATH")
# if (!inherits(x, "try-error")) {
#   x = sub("bin/python", "lib", x)
#   Sys.setenv("LD_LIBRARY_PATH" = paste0(x, ":", ld))
# }
py_gc <- reticulate::import("gc")
pandas = reticulate::py_module_available("pandas")
if (!pandas) {
  py = reticulate:::py_exe()
  cmd = paste0(py, " -m pip install pandas")
  system(cmd)
  # reticulate::py_install("pandas")
}
#
# if (!Sys.info()[["user"]] %in% c("rstudio", "jupyter")) {
#   reticulate::py_config()
#   conda <- reticulate::conda_binary()
#   cmd = paste0(conda, " update libgcc")
#   system(cmd)
#   cmd = paste0(conda, " update libstdcxx-ng")
#   system(cmd)
#   cmd = "apt-get install -y libstdc++6"
#   system(cmd)
# }
# reticulate::source_python("signal_nil_check.py")


if (!file.exists("wide.rds")) {
  wide = get_wide_data()
  readr::write_rds(wide, "wide.rds")
} else {
  wide = readr::read_rds("wide.rds")
}

# data = wide[ wide$version %in% version, ]
nfolds = 100
wide = make_folds(wide, nfolds)
if ("counts" %in% colnames(wide)) {
  wide$run = !is.na(wide$counts)
} else {
  wide$run = FALSE
}
# wide$run = in_bqt(data_bqt, wide$id) && in_bqt(measure_1min_bqt, wide$id)
if (any(!wide$run)) {
  wide %>%
    count(run,fold) %>%
    spread(run, n)
} else {
  print("All have been run")
}


fold = as.numeric(Sys.getenv("TASK_ID"))
if (!is.na(fold)) {
  print(paste0("Fold is: ", fold))
  wide = wide[ wide$fold %in% fold, ]
}


# Grab only missing data
wide = wide[!wide$run,]
iid = 1
idf = wide[iid,]
for (iid in seq(nrow(wide))) {
  idf = wide[iid,]
  print(idf)
  id = idf$id
  version = idf$version
  counts_file = file.path(version, "counts", paste0(id, ".csv.gz"))
  if (!gcs_file_exists(counts_file)) {
    out = try({
      process_nhanes_80hz(id, version, verbose = TRUE)
    })
    if (!inherits(out, "try-error")) {
      gcs_upload_file(out$csv_file)
      gcs_upload_file(out$measures_file)
      gcs_upload_file(out$meta_file)
      gcs_upload_file(out$log_file)
      gcs_upload_file(out$counts_file)
    }
  }
  py_gc$collect()

  gc(full = TRUE)
}

