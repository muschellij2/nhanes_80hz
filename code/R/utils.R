normalize_table_name = function(nh_table) {
  table = toupper(nh_table)
  if (grepl("^Y", table)) {
    table = sub("^Y_(.*)", "\\1_Y", table)
  }
  table
}

nh_table_name = function(table) {
  nh_table = toupper(table)
  if (grepl("^_Y", nh_table)) {
    nh_table = sub("^(.*)_Y", "Y_\\1", nh_table)
  }
  nh_table
}


nhanes_xpt_url = function(nh_table) {
  nh_table = nh_table_name(nh_table)
  table = normalize_table_name(nh_table)
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
  nh_table = nh_table_name(nh_table)
  table = normalize_table_name(nh_table)
  nh_tab = tolower(table)
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
  outfile = normalize_table_name(nh_table)
  outfile = paste0(outfile, ".XPT")
  file = file.path(data_dir, outfile)
  if (!file.exists(file)) {
    curl::curl_download(url, destfile = file, quiet = FALSE)
  }
  file
}
