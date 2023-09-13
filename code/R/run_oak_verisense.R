#!/usr/bin/env Rscript
library(dplyr)
library(walking)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = Sys.getenv("SGE_TASK_ID")
ifold = as.numeric(ifold)
print(paste0("fold is: ", ifold))
if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}


max_n = nrow(df)
index = 1
# for (index in seq(max_n)) {
  # print(index)
  idf = df[index,]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  files = c(
    idf$csv_file,
    idf$csv15_file,
    idf$csv10_file
  )

  outfiles = c(
    idf$oak_file,
    idf$verisense_file
  )
  sapply(dirname(outfiles), dir.create, showWarnings = FALSE, recursive = TRUE)
  get_walking = function(df, sample_rate = 80L) {
    df = walking::standardise_data(df)
    oak = estimate_steps_forest(df)
    message("OAK completed")
    vs = estimate_steps_verisense(
      df,
      sample_rate = sample_rate,
      method = "original")
    message("Verisense completed")
    vs_revised = estimate_steps_verisense(
      df,
      sample_rate = sample_rate,
      method = "revised")
    message("Verisense revised completed")
    out = list(
      oak = oak,
      vs = vs,
      vs_revised = vs_revised
    )
    out = lapply(out, function(r) {
      cn = colnames(r)
      cn[cn == "steps"] = paste0("steps_", sample_rate)
      colnames(r) = cn
      r
    })
  }

  if (!all(file.exists(outfiles)) && all(file.exists(files))) {
    df = read_80hz(idf$csv_file, progress = FALSE)
    out = get_walking(df, sample_rate = 80L)

    df = read_80hz(idf$csv15_file, progress = FALSE)
    out15 = get_walking(df, sample_rate = 15L)

    df = read_80hz(idf$csv10_file, progress = FALSE)
    out10 = get_walking(df, sample_rate = 10L)
    renamer = function(df, prefix) {
      cn = colnames(df)
      timecol = cn %in% "time"
      cn[!timecol] = paste0(prefix, cn[!timecol])
      colnames(df) = cn
      df
    }
    nn = c("oak", "vs", "vs_revised")
    result = lapply(nn, function(x) {
      prefix = paste0(x, "_")
      df = out[[x]] %>% renamer(prefix)
      df10 = out10[[x]] %>% renamer(prefix)
      df15 = out15[[x]] %>% renamer(prefix)
      res = dplyr::left_join(df, df10)
      res = dplyr::left_join(res, df15)
      res$time = c(unlist(res$time))
      res
    })
    names(result) = nn
    result$verisense = dplyr::left_join(result$vs, result$vs_revised)
    result = lapply(result, dplyr::as_tibble)
    write_csv_gz(df = result$oak, file = idf$oak_file)
    write_csv_gz(df = result$verisense, file = idf$verisense_file)
  }
# }


