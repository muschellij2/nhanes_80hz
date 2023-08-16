library(rvest)
library(curl)
library(dplyr)
library(here)
source("code/R/utils.R")
n_folds = 55

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

get_version_filenames = function(nhanes_version) {
  tarball_file = filename = NULL
  rm(list = c("filename", "tarball_file"))
  nhanes_version = tolower(nhanes_version)
  table = tolower(normalize_table_name(nhanes_version))
  suffix = ifelse(grepl("paxlux", nhanes_version), "_lux",
                  "")
  data_dir = here::here(
    "data",
    paste0("raw", suffix)
  )
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
  }
  file = file.path(data_dir, paste0(table, "_index.html"))
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
  df = dplyr::as_tibble(df)
  df = df %>%
    dplyr::filter(tools::file_ext(filename) %in% "bz2")

  df = df %>%
    mutate(day = sub(",", "", day),
           day = trimws(day))

  folder_name = paste0("pax_", sub(".*_(.*)", "\\1", table))

  make_csv_name = function(dir, folder_name, id) {
    here::here("data", dir, folder_name, paste0(id, ".csv.gz"))
  }
  df = df %>%
    dplyr::mutate(
      day_of_week = sub(",$", "", day_of_week),
      url = paste0(base_url, hrefs),
      tarball_file = here::here("data",
                                paste0("raw", suffix),
                                folder_name, filename),
      # 2x because tar.bz2
      id = tools::file_path_sans_ext(filename),
      id = tools::file_path_sans_ext(id)
    )
  df = df %>%
    dplyr::mutate(
      meta_file = make_csv_name(paste0("meta", suffix),
                                folder_name, id),
      summary_meta = make_csv_name(paste0("summary_meta", suffix),
                                   folder_name, id),
      log_file = make_csv_name(paste0("logs", suffix),
                               folder_name, id),
      csv_file = make_csv_name(paste0("csv", suffix),
                               folder_name, id),
      csv15_file = make_csv_name(paste0("csv", suffix),
                               folder_name, id),
      csv10_file = make_csv_name(paste0("csv", suffix),
                               folder_name, id),
      csv100_file = make_csv_name(paste0("csv", suffix),
                                 folder_name, id)
    )
  df = df %>%
    dplyr::mutate(version = folder_name) %>%
    dplyr::select(version, id, url, tarball_file, filename,
                  dplyr::everything())
  readr::write_rds(df, file.path(data_dir,
                                 paste0(folder_name, suffix,
                                        "_filenames.rds")))

  writeLines(df$id, file.path(data_dir, paste0(folder_name, suffix,
                                               "_ids.txt")))
  return(df)
}


get_version_filenames("pax_g")
get_version_filenames("pax_h")
get_version_filenames("pax_y")

get_version_filenames("y_paxlux")

dfs = lapply(c("pax_h", "pax_g", "pax_y"), function(version) {
  readr::read_rds(
    here::here("data", "raw", paste0(version, "_filenames.rds"))
  )
})
df = dplyr::bind_rows(dfs)

df = df %>%
  mutate(fold = seq(dplyr::n()),
         fold = floor(fold / ceiling(dplyr::n()/n_folds) + 1))
readr::write_rds(df, here::here("data", "raw", "all_filenames.rds"))




