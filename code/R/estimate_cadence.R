#!/usr/bin/env Rscript
library(dplyr)
library(walking)
# needed because
# https://github.com/OverLordGoldDragon/ssqueezepy#gpu--cpu-acceleration
Sys.setenv("SSQ_PARALLEL" = 0)
options(digits.secs = 3)
source(here::here("code", "R", "helper_functions.R"))
source(here::here("code", "R", "utils.R"))
fold = NULL
rm(list = c("fold"))

df = readRDS(here::here("data", "raw", "all_filenames.rds"))
xdf = df

ifold = get_fold()

if (!is.na(ifold)) {
  df = df %>%
    dplyr::filter(fold %in% ifold)
}


max_n = nrow(df)
index = 1
for (index in seq(max_n)) {

  idf = df[index,]
  print(paste0(index, " of ", max_n))
  print(idf$csv_file)

  files = c(
    idf$oak_file,
    idf$verisense_file
  )

  if (all(file.exists(files))) {
    data = readr::read_csv(idf$oak_file, progress = FALSE)
    long = data %>%
      tidyr::pivot_longer(cols = -time) %>%
      mutate(
        sample_rate = sub(".*_(\\d*)$", "\\1", name) %>% as.numeric()
      ) %>%
      rename(steps = value)


    out = get_walking(data, sample_rate = 80L)
    rm(data)

    data = read_80hz(idf$csv15_file, progress = FALSE)
    out15 = get_walking(data, sample_rate = 15L)
    rm(data)

    data = read_80hz(idf$csv10_file, progress = FALSE)
    out10 = get_walking(data, sample_rate = 10L)
    rm(data)

    renamer = function(data, prefix) {
      cn = colnames(data)
      timecol = cn %in% "time"
      cn[!timecol] = paste0(prefix, cn[!timecol])
      colnames(data) = cn
      data
    }
    nn = c("oak", "vs", "vs_revised")
    result = lapply(nn, function(x) {
      prefix = paste0(x, "_")
      data = out[[x]] %>% renamer(prefix)
      data10 = out10[[x]] %>% renamer(prefix)
      data15 = out15[[x]] %>% renamer(prefix)
      res = dplyr::left_join(data, data10)
      res = dplyr::left_join(res, data15)
      res$time = c(unlist(res$time))
      res
    })
    names(result) = nn
    result$verisense = dplyr::left_join(result$vs, result$vs_revised)
    result = lapply(result, dplyr::as_tibble)
    write_csv_gz(df = result$oak, file = idf$oak_file)
    write_csv_gz(df = result$verisense, file = idf$verisense_file)

    rm(data)
    rm(result)
  }
}


