get_fold = function() {
  ifold = as.numeric(Sys.getenv("SGE_TASK_ID"))
  if (is.na(ifold)) {
    ifold = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
  }
  print(paste0("fold is: ", ifold))
  ifold
}
normalize_table_name = function(nh_table) {
  table = toupper(nh_table)

  y_table = grepl("^Y_", table)
  if (any(y_table)) {
    table[y_table] = sub("^Y_(.*)", "\\1_Y", table[y_table])
  }
  table
}

nh_table_name = function(table) {
  nh_table = toupper(table)
  y_table = grepl("_Y$", nh_table)
  if (any(y_table)) {
    nh_table[y_table] = sub("^(.*)_Y", "Y_\\1", nh_table[y_table])
  }
  nh_table
}

get_wave = function(nh_table) {
  norm_table = normalize_table_name(nh_table)
  sub(".*_(.*)", "\\1", tolower(norm_table))
}

nhanes_xpt_url = function(nh_table) {
  nh_table = nh_table_name(nh_table)
  table = normalize_table_name(nh_table)
  nh_year <- nhanesA:::.get_year_from_nh_table(nh_table)
  base_url = "https://wwwn.cdc.gov/Nchs/"
  y_table = grepl("^Y_", nh_table)
  url <- paste0(base_url,  "Nhanes", "/", nh_year, "/", nh_table, ".XPT")
  if (any(y_table)) {
    url[y_table] <- paste0(base_url, nh_year[y_table],
                           "/", nh_table[y_table], ".XPT")
  }
  url
}
table_to_outdir = function(nh_table) {
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
  outdir
}
get_xpt = function(nh_table) {
  nh_table = nh_table_name(nh_table)
  table = normalize_table_name(nh_table)
  outdir = table_to_outdir(nh_table)

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



daily_min_file = function(nh_table) {
  nh_table = nh_table_name(nh_table)
  outdir = table_to_outdir(nh_table)
  stopifnot(!is.na(outdir))
  wave = get_wave(nh_table)
  pax_name = paste0("pax_", wave)

  data_dir = here::here("data", outdir)
  # dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

  outfile = normalize_table_name(nh_table)
  outfile = paste0(outfile, ".XPT")
  file = file.path(data_dir, outfile)
  file
}
read_daily_min = function(nh_table, ...) {
  file = daily_min_file(nh_table)
  if (file.exists(file)) {
    df = haven::read_xpt(file, ...)
  }
  df
}


