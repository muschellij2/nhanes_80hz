library(containerit)
library(trailrun)
debugonce(trailrun::build_renv_trigger)
trailrun::build_renv_trigger(
  includedFiles = c("renv.lock", "cloudbuild_process.yaml",
                    "Dockerfile_process"),
  yaml_filename = "cloudbuild_process.yaml",
  name = "nhanes_80hz_process"
)

image = "gcr.io/google.com/cloudsdktool/cloud-sdk:latest"
instructions = c(
  Run_shell("apt-get install -y pigz lbzip2")
)
# instructions = c(
#   Run_shell("apk --no-cached add pigz lbzip2")
# )
docker = containerit::dockerfile(
  from = clean_session(),
  image = image,
  maintainer = "StreamlineDataScience",
  instructions = instructions,
  cmd = NULL,
  container_workdir = NULL
)
dockerfile = "Dockerfile_parallel_zip"
containerit::write(docker, file = dockerfile)
tfile = file.path(tempfile(), "parallel-zip")
dir.create(tfile, showWarnings = FALSE, recursive = TRUE)
file.copy(dockerfile, file.path(tfile, "Dockerfile"))
googleCloudRunner::cr_deploy_docker(
  local = tfile,
  image_name = "cloud-sdk-zip"
)
