library(rvest)
library(curl)
library(dplyr)
library(here)

remove_double_space = function(x) {
  gsub("\\s+", " ", x)
}

read_html_newline = function(file) {
  xx = readLines(file)
  bad_string = "NEWLINE"
  xx = gsub("<br>", bad_string, xx)
  tfile = tempfile(fileext = ".html")
  writeLines(xx, tfile)
  doc = read_html(tfile)
}
nhanes_version = "pax_h"

get_version_filenames = function(nhanes_version) {
  filename = NULL
  rm(list=ls())
  data_dir = here::here("data", "raw", nhanes_version)
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
  }
  file = file.path(data_dir, paste0(nhanes_version, "_index.html"))
  base_url = "https://ftp.cdc.gov"
  url = paste0(base_url, "/pub/", nhanes_version, "/")
  if (!file.exists(file)) {
    curl::curl_download(url, destfile = file)
  }
  doc = read_html(file)


  hrefs = read_html_newline(file) %>%
    html_nodes("a") %>%
    html_attr("href")
  hrefs = hrefs[tools::file_ext(hrefs) %in% "bz2"]
  stubs = unique(dirname(hrefs))
  stopifnot(length(stubs) == 1)

  stopifnot(stubs == paste0("/pub/", nhanes_version))

  doc = read_html_newline(file) %>%
    html_nodes("pre")
  x = doc %>%  html_text()
  x = gsub("NEWLINE", "\n", x)
  x = strsplit(x, "\n")[[1]]
  x = remove_double_space(trimws(x))
  keep = !x %in% "" & !grepl("Parent", x)
  x = x[keep]
  df = strsplit(x, " ")
  df = lapply(df, trimws)
  df = do.call(rbind, df)
  colnames(df) = c("day_of_week", "month", "day", "year", "time", "am_pm",
                   "bytes", "filename")
  df = as_tibble(df)
  df = df %>%
    dplyr::filter(tools::file_ext(filename) %in% "bz2")

  df = df %>%
    mutate(
      day_of_week = sub(",$", "", day_of_week),
      url = paste0(base_url, hrefs),
      file = file.path(nhanes_version, "raw", filename),
      # 2x because tar.bz2
      id = tools::file_path_sans_ext(filename),
      id = tools::file_path_sans_ext(id),
      parquet = file.path(nhanes_version, "parquet",paste0(id, ".parquet")),
      meta = file.path(nhanes_version, "meta",paste0(id, ".csv.gz")),
      logfile = file.path(nhanes_version, "logs", paste0(id, ".csv.gz")),
      full_csv = file.path(nhanes_version, "csv", paste0(id, ".csv.gz"))
    ) %>%
    select(id, url, file, filename, everything())
  readr::write_rds(df, file.path(data_dir, paste0(nhanes_version, "_filenames.rds")))

  writeLines(df$id, file.path(data_dir, paste0(nhanes_version, "_ids.txt")))
  return(df)
}


get_version_filenames("pax_h")
get_version_filenames("pax_g")
