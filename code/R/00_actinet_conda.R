library(reticulate)

reticulate::conda_create("actinet",
                         python_version = "3.10",
                         packages = c("pip")
)

reticulate::conda_install(
  "actinet",
  pip = TRUE,
  packages = "actinet==0.7.2"
)

reticulate::use_condaenv("actinet")
model_dir = here::here("actinet_models")
if (!dir.exists(model_dir)) {
  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
}


actinet::ac_download_model(
  model_path = file.path(model_dir, actinet::ac_model_filename("walmsley")),
  classifier = "walmsley"
)

actinet::ac_download_model(
  model_path = file.path(model_dir, actinet::ac_model_filename("willetts")),
  classifier = "willetts"
)
