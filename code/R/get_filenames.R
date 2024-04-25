library(rvest)
library(curl)
library(dplyr)
library(here)
source(here::here("code/R/utils.R"))
n_folds = 200

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

  # get the full URL
  file = file.path(data_dir, paste0(table, "_index.html"))
  base_url = "https://ftp.cdc.gov"
  url = paste0(base_url, "/pub/", nhanes_version, "/")

  # download the HTML if not already here (or new clone of repo)
  if (!file.exists(file)) {
    curl::curl_download(url, destfile = file)
  }
  # read the HTML as a Document we can use some XPATH/CSS to query
  doc = read_html(file)

  # Give me the links!
  hrefs = read_html_newline(file) %>%
    html_nodes("a") %>%
    html_attr("href")
  # Only relevant links
  hrefs = hrefs[tools::file_ext(hrefs) %in% "bz2"]
  stubs = unique(dirname(hrefs))
  stopifnot(length(stubs) == 1)

  stopifnot(stubs == paste0("/pub/", nhanes_version))

  # read in the table of all the files available
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

      counts_1s_file = make_csv_name(paste0("counts_1s", suffix),
                                     folder_name, id),
      counts_60s_file = make_csv_name(paste0("counts_60s", suffix),
                                      folder_name, id),


      counts_file = make_csv_name(paste0("counts", suffix),
                                  folder_name, id),

      csv_file = make_csv_name(paste0("csv", suffix),
                               folder_name, id),

      calibrated_file = make_csv_name(paste0("gravity_calibrated", suffix),
                               folder_name, id),

      calibration_params_file = make_csv_name(paste0("gravity_params", suffix),
                                      folder_name, id),

      ggir_calibrated_file = make_csv_name(paste0("ggir_gravity_calibrated", suffix),
                                      folder_name, id),

      ggir_calibration_params_file = make_csv_name(paste0("ggir_gravity_params", suffix),
                                      folder_name, id),

      acc_csv_file = make_csv_name(paste0("acc_csv", suffix),
                                   folder_name, id),

      csv15_file = make_csv_name(paste0("csv_15", suffix),
                                 folder_name, id),
      csv10_file = make_csv_name(paste0("csv_10", suffix),
                                 folder_name, id),
      csv30_file = make_csv_name(paste0("csv_30", suffix),
                                 folder_name, id),

      csv08_file = make_csv_name(paste0("csv_08", suffix),
                                 folder_name, id),
      csv16_file = make_csv_name(paste0("csv_16", suffix),
                                 folder_name, id),
      csv32_file = make_csv_name(paste0("csv_32", suffix),
                                 folder_name, id),
      csv64_file = make_csv_name(paste0("csv_64", suffix),
                                 folder_name, id),
      csv100_file = make_csv_name(paste0("csv_100", suffix),
                                  folder_name, id)
    )

  df = df %>%
    dplyr::mutate(
      oak_file = make_csv_name(paste0("oak", suffix),
                               folder_name, id),
      verisense_file = make_csv_name(paste0("verisense", suffix),
                                     folder_name, id),
      adept_file = make_csv_name(paste0("adept", suffix),
                                 folder_name, id)
    )

  df = df %>%
    dplyr::mutate(
      stepcount_file = make_csv_name(paste0("stepcount", suffix),
                               folder_name, id),
      stepcount_params_file = make_csv_name(paste0("stepcount_params", suffix),
                                     folder_name, id),
      stepcount30_file = make_csv_name(paste0("stepcount_30", suffix),
                                     folder_name, id),
      stepcount30_params_file = make_csv_name(paste0("stepcount_30_params", suffix),
                                            folder_name, id)
    )

  df = df %>%
    dplyr::mutate(
      rf_stepcount_file = make_csv_name(paste0("rf_stepcount", suffix),
                                     folder_name, id),
      rf_stepcount_params_file = make_csv_name(paste0("rf_stepcount_params", suffix),
                                            folder_name, id),
      rf_stepcount30_file = make_csv_name(paste0("rf_stepcount_30", suffix),
                                       folder_name, id),
      rf_stepcount30_params_file = make_csv_name(paste0("rf_stepcount30_params", suffix),
                                               folder_name, id),
    )

  df = df %>%
    dplyr::mutate(
      measures_file = make_csv_name(paste0("measures", suffix),
                                        folder_name, id)
    )

  df = df %>%
    dplyr::mutate(
      acc_steps_1s_file = make_csv_name(paste0("acc_steps_1s", suffix),
                                     folder_name, paste0(id, "1sec")),
      steps_1s_file = make_csv_name(paste0("steps_1s", suffix),
                                     folder_name, id)
    )


  df = df %>%
    dplyr::mutate(
      nonwear_swan_file = make_csv_name(paste0("nonwear_swan", suffix),
                                        folder_name, id),
      nonwear_weartime_file = make_csv_name(paste0("nonwear_weartime", suffix),
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




