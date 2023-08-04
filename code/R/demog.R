library(haven)
library(dplyr)
library(nhanesA)
library(rlang)
source("code/R/utils.R")
table = "DEMO_G"

process_demog = function(df) {
  if ("full_sample_2_year_mec_exam_weight" %in% colnames(df)) {
    atr = attributes(df$full_sample_2_year_mec_exam_weight)
    df = df %>%
      dplyr::mutate(
        full_sample_2_year_mec_exam_weight =
          dplyr::if_else(full_sample_2_year_mec_exam_weight == 0,
                         NA_real_,
                         full_sample_2_year_mec_exam_weight)
      )
    attributes(df$full_sample_2_year_mec_exam_weight) = atr
  }
  if ("age_in_years_at_screening" %in% colnames(df) &&
      is.numeric(df$age_in_years_at_screening)) {
    atr = attributes(df$age_in_years_at_screening)
    df = df %>%
      dplyr::mutate(
        age_in_years_at_screening =
          ifelse(age_in_years_at_screening >= 80,
                 "80 years of age and over",
                 age_in_years_at_screening)
      )
    attributes(df$age_in_years_at_screening) = atr
  }
  df
}
read_and_relabel = function(table, ...) {
  file = here::here("data", "demographics", paste0(table, ".XPT"))
  df = haven::read_xpt(file, ...)
  table = normalize_table_name(table)
  nh_table = nh_table_name(table)
  translations = nhanesA::nhanesTranslate(
    nh_table = nh_table, data = NULL,
    colnames = colnames(df),
    nchar = 100)
  df = nhanesA::nhanesTranslate(nh_table = nh_table, data = df,
                                colnames = colnames(df),
                                nchar = 100)
  if (nh_table == "DEMO_G") {
    recode_vars = c("DMDHHSIZ", "DMDFMSIZ", "DMDHHSZA",
                    "DMDHHSZB", "DMDHHSZE")
    # cross coding!
    df = nhanesA::nhanesTranslate(
      nh_table = "DEMO_H",
      data = df,
      colnames = recode_vars,
      nchar = 100)
    translations[names(translations) %in% recode_vars] = NULL
    tt = nhanesA::nhanesTranslate(
      nh_table = "DEMO_H",
      data = df,
      colnames = recode_vars,
      nchar = 100)
    translations = c(translations, tt)
  }

  cn = colnames(df)
  for (icn in cn) {
    attr(df[[icn]], "variable_name") = icn
    attr(df[[icn]], "translation") = translations[[icn]]
  }
  cn = sapply(df, attr, "label")
  cn = janitor::make_clean_names(cn)
  colnames(df) = cn

  norm_table = normalize_table_name(nh_table)
  df = process_demog(df)
  df = df %>%
    dplyr::mutate(
      nh_table = table,
      table = norm_table,
      wave = sub(".*_(.*)", "\\1", tolower(norm_table)),
      version = paste0("pax_", wave)
    )
  df = tibble::as_tibble(df)
  df
}
waves = c("DEMO_G", "DEMO_H", "DEMO_Y")
dfs = lapply(waves, read_and_relabel)
names(dfs) = waves
label_df = function(df) {
  tibble::tibble(
    new = colnames(df),
    original = sapply(df, function(x) {
      attr(x, "variable_name") %||% NA_character_
    })
  )
}
get_col_from_label = function(df, label) {
  labels = sapply(df, attr, "label")
  df[, labels %in% label]
}
dplyr::bind_rows(dfs)
