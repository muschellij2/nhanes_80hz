
make_parallel_steps = function(version, ids, file_to_run = "make_csvs.sh",
                               nfolds = 32,
                               folds_to_run = NULL) {
  fold_id_file = paste0(version, "_folds.txt")

  n = length(ids)
  folds = rep(1:nfolds, each = ceiling(n/nfolds))[1:length(ids)]
  stopifnot(!anyNA(folds))

  writeLines(paste(ids, folds), fold_id_file, sep = "\n")
  googleCloudStorageR::gcs_upload(fold_id_file, bucket = bucket,
                                  name = fold_id_file)

  googleCloudStorageR::gcs_upload(file_to_run, bucket = bucket,
                                  name = file_to_run)

  cr_buildstep_gsutil2 = function(...) {
    cr_buildstep(
      name = "gcr.io/streamline-resources/cloud-sdk-zip:latest",
      entrypoint = "gsutil",
      ...
    )
  }

  pre_steps = c(
    cr_buildstep_gsutil2(
      c("cp",paste0("gs://", bucket, "/", file_to_run), file_to_run),
      id = "copier"
    ),
    cr_buildstep_gsutil2(
      c("version", "-l"),
      id = "gsutil_version"
    ),
    cr_buildstep_gsutil2(
      c("cp",paste0("gs://", bucket, "/", fold_id_file), fold_id_file),
      id = "copy_ids"
    ),
    cr_buildstep_cat(
      file_to_run,
      id = "print_file",
      waitFor = "copier"
    ),
    cr_buildstep_cat(
      fold_id_file,
      id = "print_ids",
      waitFor = "copy_ids"
    )
  )

  if (is.null(folds_to_run)) {
    folds_to_run = unique(folds)
  }

  all_steps = lapply(folds_to_run, function(r) {
    steps = cr_buildstep(
      name = "gcr.io/streamline-resources/cloud-sdk-zip:latest",
      entrypoint = "bash",
      args = file_to_run,
      env = c(paste0("fold=", r),
              paste0("version=", version)),
      waitFor = "copy_ids"
    )
  })
  all_steps = unname(all_steps)
  steps = sapply(all_steps, c)

  # steps = all_steps[[5]]
  steps = c(pre_steps, steps)
  names(steps) = NULL
  steps
}





run_ids = function(version,
                   ids,
                   n_same_time = 4,
                   file_to_run = "make_csvs.sh") {
  n = length(ids)

  googleCloudStorageR::gcs_upload(file_to_run, bucket = bucket,
                                  name = file_to_run)
  cr_buildstep_gsutil2 = function(...) {
    cr_buildstep(
      name = "gcr.io/streamline-resources/cloud-sdk-zip:latest",
      entrypoint = "gsutil",
      ...
    )
  }

  pre_steps = c(
    cr_buildstep_gsutil2(
      c("cp",paste0("gs://", bucket, "/", file_to_run), file_to_run),
      id = "copier"
    ),
    cr_buildstep_gsutil2(
      c("version", "-l"),
      id = "gsutil_version"
    ),
    # cr_buildstep_gsutil2(
    #   c("cp",paste0("gs://", bucket, "/", fold_id_file), fold_id_file),
    #   id = "copy_ids"
    # ),
    cr_buildstep_cat(
      file_to_run,
      id = "print_file",
      waitFor = "copier"
    )
    # cr_buildstep_cat(
    #   fold_id_file,
    #   id = "print_ids",
    #   waitFor = "copy_ids"
    # )
  )

  iteration = rep(1:n_same_time, length = n * n_same_time)[1:n]
  fold = ceiling(seq(n)/n_same_time)
  run_ids = paste0("round_", fold, "_", iteration)
  wait_fors = paste0("round_", fold - 1, "_", iteration)
  wait_fors[!wait_fors %in% run_ids] = "copier"

  all_steps = mapply(function(id, run_id, wait_for) {
    steps = cr_buildstep(
      name = "gcr.io/streamline-resources/cloud-sdk-zip:latest",
      entrypoint = "bash",
      args = file_to_run,
      env = c(paste0("ids=", id),
              paste0("version=", version)),
      id = run_id,
      waitFor = wait_for
    )
  }, SIMPLIFY = FALSE, ids, run_ids, wait_fors)
  all_steps = unname(all_steps)
  steps = sapply(all_steps, c)

  # steps = all_steps[[5]]
  steps = c(pre_steps, steps)
  names(steps) = NULL
  steps
}


get_bqt_ids = function(table, dataset, project) {
  bqt = bq_table(project = project,
                 dataset = dataset,
                 table = table)

  bqd = bigrquery::bq_dataset(bqt$project, bqt$dataset)
  dataset_exists = bigrquery::bq_dataset_exists(bqd)
  if (!dataset_exists) {
    bigrquery::bq_dataset_create(bqd)
  }

  if (!bigrquery::bq_table_exists(bqt)) {
    curr_ids = NULL
  } else {
    curr_ids = bigrquery::bq_table_download(bqt)
    curr_ids = curr_ids$id
  }
  list(bqt = bqt, curr_ids = curr_ids)
}
