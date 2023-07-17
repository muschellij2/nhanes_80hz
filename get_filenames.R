library(rvest)
library(curl)
library(dplyr)

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

get_version_filenames = function(version) {
  data_dir = here::here("raw", version)
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
  }
  file = paste0(version, "_index.html")
  base_url = "https://ftp.cdc.gov"
  url = paste0(base_url, "/pub/", version, "/")
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

  stopifnot(stubs == paste0("/pub/", version))

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
    filter(tools::file_ext(filename) %in% "bz2")

  df = df %>%
    mutate(
      day_of_week = sub(",$", "", day_of_week),
      url = paste0(base_url, hrefs),
      file = file.path(version, "raw", filename),
      # 2x because tar.bz2
      id = tools::file_path_sans_ext(filename),
      id = tools::file_path_sans_ext(id),
      parquet = file.path(version, "parquet",paste0(id, ".parquet")),
      meta = file.path(version, "meta",paste0(id, ".csv.gz")),
      logfile = file.path(version, "logs", paste0(id, ".csv.gz")),
      full_csv = file.path(version, "csv", paste0(id, ".csv.gz"))
    ) %>%
    select(id, url, file, filename, everything())
  readr::write_rds(df, paste0(version, "_filenames.rds"))

  writeLines(df$id, paste0(version, "_ids.txt"))
  return(df)
}


get_version_filenames("pax_h")
get_version_filenames("pax_g")
