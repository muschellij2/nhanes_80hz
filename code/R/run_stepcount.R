library(janitor)
library(tibble)
library(stepcount)
unset_reticulate_python()
use_stepcount_condaenv()
library(dplyr)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

model_path_by_type = function(model_type) {
  model_filename = sc_model_filename(model_type)
  model_path = switch(
    model_type,
    ssl = model_filename,
    rf = paste0("rf-", model_filename))
  model_path = here::here("stepcount_models", model_path)
  if (!file.exists(model_path)) {
    model_path = NULL
  }
  model_path
}

# model_types = c("ssl", "rf")
model_types = "ssl"
sample_rate = 80L
csv_col = ifelse(sample_rate == 80, "csv_file",
                 paste0("csv", sample_rate, "_file"))


stepcount_cols = sapply(model_types, function(model_type) {
  stepcount_col = ifelse(sample_rate == 80, "stepcount_file",
                         paste0("stepcount", sample_rate, "_file"))
  stepcount_col = ifelse(model_type != "ssl",
                         paste0(model_type, "_", stepcount_col),
                         stepcount_col)
  stepcount_col
})

# Load the models
models = lapply(model_types, function(model_type) {
  model_path = model_path_by_type(model_type)
  res = sc_load_model(model_type = model_type,
                      model_path = model_path,
                      as_python = TRUE)
  res
})
names(models) = model_types
ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}

i = 1
model_type = model_types[1]

for (i in seq_len(nrow(df))) {
  idf = df[i,]
  print(paste0(i, " of ", nrow(df)))
  file = idf[[csv_col]]
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  print(file)
  stepcount_outfiles = unlist(idf[,stepcount_cols])
  names(stepcount_outfiles) = model_types
  if (!all(file.exists(stepcount_outfiles))) {
    run_file = rename_xyzt(file, tmpdir = tempdir())

    for (model_type in model_types) {
      message("Running model: ", model_type)
      model = models[[model_type]]
      stepcount_col = stepcount_cols[model_type]
      if (!file.exists(idf[[stepcount_col]])) {
        out = stepcount_with_model(file = run_file,
                                   model = model,
                                   model_type = model_type)
        info = tibble::as_tibble(out$info)
        info = janitor::clean_names(info)
        info$filename = file

        stopifnot(all(out$walking$walking %in% c(NaN, 0L, 1L)))
        result = dplyr::full_join(out$steps, out$walking)
        result = result %>%
          dplyr::mutate(non_wear = is.na(steps) & is.na(walking),
                        walking = walking > 0)
        write_csv_gz(result, idf[[stepcount_col]])
        rm(out)
        rm(result)
      }
    }
    suppressWarnings({
      file.remove(run_file)
      file.remove(paste0(run_file, "bak"))
    })
  }
}
