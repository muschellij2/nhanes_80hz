library(haven)
library(dplyr)
library(nhanesA)
library(rlang)
source(here::here("code/R/utils.R"))
source(here::here("code/R/helper_functions.R"))

nh_tables = c("PAXHD_G", "PAXHD_H", "Y_PAXHD")
nh_table = nh_tables[1]
for (nh_table in nh_tables) {
  table = normalize_table_name(nh_table)
  file = here::here("data", "raw", paste0(table, ".XPT"))
  if (!file.exists(file)) {
    url = nhanes_xpt_url(nh_table)
    curl::curl_download(url, file)
  }
  df = haven::read_xpt(file)
  df = tibble::as_tibble(df)

  cn_to_translate = setdiff(colnames(df), c("PAXFTIME", "PAXETLDY"))
  translations = nhanesA::nhanesTranslate(
    nh_table = nh_table,
    data = NULL,
    colnames = cn_to_translate,
    nchar = 100L)


  df = nhanesA::nhanesTranslate(nh_table = nh_table,
                                data = df,
                                colnames = cn_to_translate,
                                nchar = 100L)
  empty_to_na = function(x) {
    x[x %in% ""] = NA_character_
    x
  }
  df = df %>%
    dplyr::mutate(
      dplyr::across(
        c(PAXSENID, PAXFDAY, PAXLDAY,
          PAXFTIME, PAXETLDY, PAXHAND, PAXORENT),
        empty_to_na)
    )
  outfile = here::here("data", "raw", paste0(table, ".csv.gz"))
  write_csv_gz(df, outfile)
}
