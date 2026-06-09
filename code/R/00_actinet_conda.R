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
