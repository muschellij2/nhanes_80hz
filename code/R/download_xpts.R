library(rvest)
library(curl)
library(dplyr)
library(here)


nhanes_xpt_url = function(nh_table) {
  nh_year <- nhanesA:::.get_year_from_nh_table(nh_table)
  base_url = "https://wwwn.cdc.gov/Nchs/"
  if (grepl("^Y_", nh_table)) {
    url <- paste0(base_url, nh_year, "/", nh_table, ".XPT")
  } else {
    url <- paste0(base_url,  "Nhanes", "/", nh_year, "/", nh_table, ".XPT")
  }
  url
}
get_xpt = function(nh_table) {
  nh_tab = tolower(nh_table)
  nh_tab = sub("y_(.*)", "\\1_y", nh_tab)
  outdir = dplyr::case_when(
    grepl("paxmin_", nh_tab) ~ "raw_min",
    grepl("paxday_", nh_tab) ~ "raw_day",
    grepl("pax_", nh_tab) ~ "raw",
    grepl("^demo", nh_tab) ~ "demographics",
    TRUE ~ NA_character_
  )
  stopifnot(!is.na(outdir))
  data_dir = here::here("data", outdir)
  dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

  url = nhanes_xpt_url(nh_table)
  outfile = sub("^Y_(.*)", "\\1_Y", nh_table)
  outfile = paste0(outfile, ".XPT")
  file = file.path(data_dir, outfile)
  if (!file.exists(file)) {
    curl::curl_download(url, destfile = file, quiet = FALSE)
  }
  file
}

get_xpt(nh_table = "Y_PAXMIN")
get_xpt(nh_table = "PAXMIN_G")
get_xpt(nh_table = "PAXMIN_H")
